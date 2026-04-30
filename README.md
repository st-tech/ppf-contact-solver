# ZOZO's Contact Solver 🫶

A contact solver for physics-based simulations
involving 👚 shells, 🪵 solids and 🪢 rods. All made by [ZOZO, Inc.](https://corp.zozo.com/en/), the largest fashion e-commerce company in Japan.

[![Getting Started](https://github.com/st-tech/ppf-contact-solver/actions/workflows/getting-started.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/getting-started.yml)
[![All Examples](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once.yml)
[![All Examples (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once-win.yml)
[![Python API Docs](https://github.com/st-tech/ppf-contact-solver/actions/workflows/make-docs.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/make-docs.yml)
[![Docker Build](https://github.com/st-tech/ppf-contact-solver/actions/workflows/build-docker.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/build-docker.yml)
[![Build Windows](https://github.com/st-tech/ppf-contact-solver/actions/workflows/release-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/release-win.yml)
[![Blender CI](https://github.com/st-tech/ppf-contact-solver/actions/workflows/blender.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/blender.yml)
![solver_logo](./asset/image/teaser-image.jpg)

> 🤖 We **highly** respect that readers expect to hear the author's original voice and tone, which we work to retain throughout. Our use of LLMs is clarified in [(Markdown)](./articles/llm_transparency.md).

## 👀 Quick Look

🎨 Simulate remotely from our [Blender add-on](https://st-tech.github.io/ppf-contact-solver) (screenshots taken on macOS; you can also run locally if you have a modern NVIDIA GPU on Windows or Linux)

![glance-blender](./docs/blender_addon/images/blender.webp)

🚀 Or double click `start.bat` (Windows) or run a Docker command (Linux/Windows) to get it running

![glance-terminal](./asset/image/glance-terminal.webp)

🌐 Click the URL and explore our examples

![glance-jupyter](./asset/image/glance-jupyter.webp)

## ✨ Highlights

- **💪 Robust**: Contact resolutions are penetration-free. No snagging intersections.
- **⏲ Scalable**: An extreme case includes beyond 180M contacts. Not just one million.
- **🚲 Cache Efficient**: All on the GPU runs in single precision. No double precision.
- **🥼 Not Rubbery**: Triangles never extend beyond strict upper bounds (e.g., 1%).
- **📐 Finite Element Method**: We use FEM for deformables and symbolic force jacobians.
- **⚔️ Highly Stressed**: We run GitHub Actions to run stress tests [10 times in a row](#️-ten-consecutive-runs).
- **🚀 Massively Parallel**: Both contact and elasticity solvers are run on the GPU.
- **🪟 Windows Executable**: No installation wizard shown. Just unzip and run [(Video)](https://zozo.box.com/s/9rthkw122fyss5qxuf5mie9xywg7jzdz).
- **🐳 Docker Sealed**: All can be deployed fast. The image is ~1GB.
- **🌐 JupyterLab Included**: Open your browser and run examples right away [(Video)](https://zozo.box.com/s/jgd6ijfmwee04vvnnfzapq7m2eq7cxy8).
- **🐍 Documented Python APIs**: Our Python code is fully [docstringed](https://st-tech.github.io/ppf-contact-solver/jupyterlab_api/module_reference.html) and lintable [(Video)](https://zozo.box.com/s/52atrfn70vn8u07iwzbyrrz5ameczo03).
- **☁️ Cloud-Ready**: Our solver can be seamlessly deployed on major cloud platforms.
- **🎨 Blender Add-on**: Simulate remotely and fetch results locally, even on macOS.
- **🤖 MCP Support**: Let a LLM run simulations for you using natural language.
- **✨ Stay Clean**: You can remove all traces after use.

> ⚠️ Built for offline uses; not real time. Some examples may run at an interactive rate.

## 🔖 Table of Contents

- [📝 Change History](#-change-history)
- [🎓 Technical Materials](#-technical-materials)
- [⚡️ Requirements](#️-requirements)
- [💨 Getting Started](#-getting-started)
  - [🪟 Windows Native Executable](#-windows-native-executable)
  - [🐳 Docker (Linux and Windows)](#-docker-linux-and-windows)
- [🐍 How To Use](#-how-to-use)
  - [🎨 Blender Add-on](#-blender-add-on)
  - [🌐 JupyterLab](#-jupyterlab)
    - [📚 Python APIs and Parameters](#-python-apis-and-parameters)
- [🔍 Obtaining Logs](#-obtaining-logs)
- [🖼️ Catalogue](#️-catalogue)
  - [🎨 Blender Add-on Examples](#-blender-add-on-examples)
  - [🌐 JupyterLab Examples](#-jupyterlab-examples)
  - [💰 Budget Table on AWS](#-budget-table-on-aws)
  - [🏗️ Large Scale Examples](#️-large-scale-examples)
- [🚀 GitHub Actions](#-github-actions)
  - [⚔️ Ten Consecutive Runs](#️-ten-consecutive-runs)
  - [📦 Action Artifacts](#-action-artifacts)
- [📡 Deploying on Cloud Services](#-deploying-on-cloud-services)
  - [📦 Deploying on vast.ai](#-deploying-on-vastai)
  - [📦 Deploying on RunPod](#-deploying-on-runpod)
  - [📦 Deploying on Scaleway](#-deploying-on-scaleway)
  - [📦 Deploying on Amazon Web Services](#-deploying-on-amazon-web-services)
  - [📦 Deploying on Google Compute Engine](#-deploying-on-google-compute-engine)
- [🧩 Community's Blender Add-ons](#-communitys-blender-add-ons)
- [📬 Contributing](#-contributing)
- [🙏 Acknowledgements](#-acknowledgements)

### 📚 Advanced Contents

- 🧑 Setting Up Your Development Environment [(Markdown)](./articles/develop.md#-setting-up-your-development-environment)
- 🐞 Bug Fixes and Updates [(Markdown)](./articles/bug.md)

## 📝 Change History

- (2026.04.30) Added a Blender Add-on support. See the [documentation](https://st-tech.github.io/ppf-contact-solver).
- (2025.12.18) Added native Windows standalone executable build support [(Video)](https://zozo.box.com/s/9rthkw122fyss5qxuf5mie9xywg7jzdz).
- (2025.11.26) Added [large-woven.ipynb](./examples/large-woven.ipynb) [(Video)](https://zozo.box.com/s/kc81msjfo4yw9eozn8i8bean0gbph0pj) to [large scale examples](#️-large-scale-examples).
- (2025.11.12) Added [five-twist.ipynb](./examples/five-twist.ipynb) [(Video)](https://zozo.box.com/s/36h8jpu5vcgc5t4xln2l68afj7izsx4h) and [large-five-twist.ipynb](./examples/large-five-twist.ipynb) [(Video)](https://zozo.box.com/s/v62q7cbfnpl3hufwwy2nmewyes2w1iw6) showcasing over 180M count. See [large scale examples](#️-large-scale-examples).
- (2025.10.03) Massive refactor of the codebase [(Markdown)](./articles/refactor_202510.md). Note that this change includes breaking changes to our Python APIs.
- (2025.08.09) Added a hindsight note in [eigensystem analysis](./articles/eigensys.md) to acknowledge prior work by [Poya et al. (2023)](https://romeric.github.io/).
- (2025.05.01) Simulation states now can be saved and loaded [(Video)](https://zozo.box.com/s/7v0exrbptvfli4o4z91pqtmz0tehdn62).

<details>
<summary>More history records</summary>
- (2025.04.02) Added 9 examples. See the [catalogue](#️-catalogue).
- (2025.03.03) Added a [budget table on AWS](#-budget-table-on-aws).
- (2025.02.28) Added a [reference branch and a Docker image of our TOG paper](#-technical-materials).
- (2025.02.26) Added Floating Point-Rounding Errors in ACCD in [hindsight](./articles/hindsight.md).
- (2025.02.07) Updated the [trapped example](./examples/trapped.ipynb) [(Video)](https://zozo.box.com/s/lnnyeqrvm86rxnwyjxhojfj0jgm5nphn) with squishy balls.
- (2025.03.03) Added a [budget table on AWS](#-budget-table-on-aws).
- (2025.02.28) Added a [reference branch and a Docker image of our TOG paper](#-technical-materials).
- (2025.02.26) Added Floating Point-Rounding Errors in ACCD in [hindsight](./articles/hindsight.md).
- (2025.02.07) Updated the [trapped example](./examples/trapped.ipynb) [(Video)](https://zozo.box.com/s/lnnyeqrvm86rxnwyjxhojfj0jgm5nphn) with squishy balls.
- (2025.1.8) Added a [domino example](./examples/domino.ipynb) [(Video)](https://zozo.box.com/s/p5ksfqja1ew3c6vntco5zq6g0kgf7xoo).
- (2025.1.5) Added a [single twist example](./examples/twist.ipynb) [(Video)](https://zozo.box.com/s/4phoyyeertd2mcfv436kp2ojmo1x0eio).
- (2024.12.31) Added full documentation for Python APIs, parameters, and log files [(GitHub Pages)](https://st-tech.github.io/ppf-contact-solver).
- (2024.12.27) Line search for strain limiting is improved [(Markdown)](./articles/bug.md#new-strain-limiting-line-search)
- (2024.12.23) Added [(Bug Fixes and Updates)](./articles/bug.md)
- (2024.12.21) Added a [house of cards example](./examples/cards.ipynb) [(Video)](https://zozo.box.com/s/7c114pua0107xkz4nc3bwfdzpkhgn1o9)
- (2024.12.18) Added a [frictional contact example](./examples/friction.ipynb): armadillo sliding on the slope [(Video)](https://zozo.box.com/s/15r5o7rrowwtbrsrjjpj35v8xt92ufhr)
- (2024.12.18) Added a [hindsight](./articles/hindsight.md) noting that the tilt angle was not $30^\circ$, but rather $26.57^\circ$
- (2024.12.16) Removed thrust dependencies to fix runtime errors for the driver version `560.94` [(Issue Link)](https://github.com/st-tech/ppf-contact-solver/issues/1)
</details>

## 🎓 Technical Materials

#### 📘 **A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness**

- 📚 Published in [ACM Transactions on Graphics (TOG) Vol.43, No.6](https://dl.acm.org/doi/abs/10.1145/3687908)
- 🎥 Main video [(Video)](https://zozo.box.com/s/ylyv8798ue38ra5vqo2dgjhx48edy6j9)
- 🎥 Additional video examples [(Directory)](https://zozo.box.com/s/inyyosspazk7z79zf9yeikt4xxy641g3)
- 🎥 Presentation videos [(Short)](https://zozo.box.com/s/fv07obu61ajn8s8s6ub9i7j9id5h4l2v) [(Long)](<https://zozo.box.com/s/3ewlqk47bss9v39g26nlsb99qhxsc50v>)
- 📃 Main paper [(PDF)](https://zozo.box.com/s/ckss1ejz0lbw848pg1qo7eqtvujsogy5) ([Hindsight)](./articles/hindsight.md)
- 📊 Supplementary PDF [(PDF)](https://zozo.box.com/s/ylqjobig0bq55eox5evjt49u2tcyqljm)
- 🤖 Supplementary scripts [(Directory)](https://zozo.box.com/s/m6h1t5eykdsuynx7kqr3wqi9789pfl9n)
- 🔍 Singular-value eigenanalysis [(Markdown)](./articles/eigensys.md)

##### 📌 Reference Implementation

The main branch is undergoing frequent updates and will deviate from the paper.
To retain consistency with the paper, we have created a new branch ```sigasia-2024```.

- 🛠️ Only maintenance updates are planned for this branch.
- 🚫 General users *should not* use this branch as it is not optimized for best performance.
- 🚫 All algorithmic changes listed in this [(Markdown)](./articles/bug.md) are excluded from this branch.
- 📦 We also provide a pre-compiled Docker image: ```ghcr.io/st-tech/ppf-contact-solver-compiled-sigasia-2024:latest``` of this branch.
- 🌐 [Template Link for vast.ai](https://cloud.vast.ai/?ref_id=85288&creator_id=85288&name=ppf-contact-solver-sigasia-2024)
- 🌐 [Template Link for RunPods](https://runpod.io/console/deploy?template=ooqpniuixi&ref=bhy3csxy)

## ⚡️ Requirements

- 🔥 A modern NVIDIA GPU (CUDA 12.8 or newer)
- 💻 x86 architecture (arm64 is not supported)
- 🐳 A Docker environment (see [below](#-docker-linux-and-windows)) or 🪟 Windows 10/11 for native executable (see [below](#-windows-native-executable))
- 🎨 Blender 5+ (only if you intend to use the Blender add-on)

## 💨 Getting Started

Whether you plan to use the Blender add-on or the JupyterLab interface, the solver engine itself must first be deployed. The steps below apply to both.

> ⚠️ Do not run `warmup.py` locally. If you do, you are very likely to hit failures and find it difficult to cleanup.

#### 🪟 Windows Native Executable

For Windows 10/11 users, a self-contained executable (~230MB) is available.
No Python, Docker, or CUDA Toolkit installation is needed.
All should simply work out of the box [(Video)](https://zozo.box.com/s/9rthkw122fyss5qxuf5mie9xywg7jzdz).

> 🤔 If you are cautious, you can review the [build workflow](https://github.com/st-tech/ppf-contact-solver/actions/workflows/release-win.yml) to verify safety yourself.
We try to maximize transparency; **we never build locally and upload.**

1. Install the latest NVIDIA driver [(Link)](https://www.nvidia.com/en-us/drivers/)
2. Download the latest release from [GitHub Releases](https://github.com/st-tech/ppf-contact-solver/releases) and unzip
3. Double click `start.bat`

JupyterLab frontend will auto-start. You should be able to access it at <http://localhost:8080>.

#### 🐳 Docker (Linux and Windows)

Install a NVIDIA driver [(Link)](https://www.nvidia.com/en-us/drivers/) on your host system and follow the instructions below specific to the operating system to get a Docker running:

🐧 Linux | 🪟 Windows
----|----
Install the Docker engine from here [(Link)](https://docs.docker.com/engine/install/). Also, install the NVIDIA Container Toolkit [(Link)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Just to make sure that the Container Toolkit is loaded, run `sudo service docker restart`. | Install the Docker Desktop [(Link)](https://docs.docker.com/desktop/setup/install/windows-install/). You may need to log out or reboot after the installation. After logging back in, launch Docker Desktop to ensure that Docker is running.

Next, run the following command to start the container. If no edits are needed, just copy and paste:

##### 🪟 Windows (PowerShell)

```bash
$MY_WEB_PORT = 8080      # JupyterLab port on your side
$MY_BLENDER_PORT = 9090  # Solver port for the Blender add-on
$IMAGE_NAME = "ghcr.io/st-tech/ppf-contact-solver-compiled:latest"
docker run --rm -it `
  --name ppf-contact-solver `
  --gpus all `
  -p ${MY_WEB_PORT}:${MY_WEB_PORT} `
  -p ${MY_BLENDER_PORT}:${MY_BLENDER_PORT} `
  -e WEB_PORT=${MY_WEB_PORT} `
  $IMAGE_NAME # Image size ~1GB
```

##### 🐧 Linux (Bash/Zsh)

```bash
MY_WEB_PORT=8080      # JupyterLab port on your side
MY_BLENDER_PORT=9090  # Solver port for the Blender add-on
IMAGE_NAME=ghcr.io/st-tech/ppf-contact-solver-compiled:latest
docker run --rm -it \
  --name ppf-contact-solver \
  --gpus all \
  -p ${MY_WEB_PORT}:${MY_WEB_PORT} \
  -p ${MY_BLENDER_PORT}:${MY_BLENDER_PORT} \
  -e WEB_PORT=${MY_WEB_PORT} \
  $IMAGE_NAME # Image size ~1GB
```

The image download shall be started.
Our image is hosted on [GitHub Container Registry](https://github.com/st-tech/ppf-contact-solver/pkgs/container/ppf-contact-solver-compiled) (~1GB).
JupyterLab will then auto-start.
Eventually you should be seeing:

```
==== JupyterLab Launched! 🚀 ====
     http://localhost:8080
    Press Ctrl+C to shutdown
================================
```

Next, open your browser and navigate to <http://localhost:8080>. The port `8080` can change if you change the `MY_WEB_PORT` variable.
Keep your terminal window open.
Now you are ready to go! 🎉

#### 🛑 Shutting Down

To shut down the container, just press `Ctrl+C` in the terminal.
The container will be removed and all traces will be cleaned up. 🧹

> If you wish to keep the container running in the background, replace `--rm` with `-d`. To shutdown the container and remove it, run `docker stop ppf-contact-solver && docker rm ppf-contact-solver`.

#### 🔧 Advanced Installation

If you wish to build the docker image from scratch, please refer to the cleaner installation guide [(Markdown)](./articles/install.md).

## 🐍 How To Use

We provide two frontends: a Blender add-on and a JupyterLab interface. The Blender add-on lets you build scenes and run simulations entirely within Blender's UI, while JupyterLab lets you script everything in Python from your browser. Both communicate with the same solver engine, so pick whichever you like.

In both cases, you can interact with the simulator on your laptop while the actual simulation runs on a remote headless server over the internet.
This means that **you don't have to own NVIDIA hardware**, but can rent it at [vast.ai](https://vast.ai) or [RunPod](https://www.runpod.io/) for less than $0.5 per hour.
That said, if you do have a modern NVIDIA GPU on a local Windows or Linux machine, you can also run the solver directly on it.
Actually, this [(Video)](https://zozo.box.com/s/jgd6ijfmwee04vvnnfzapq7m2eq7cxy8) was recorded on a [vast.ai](https://vast.ai) instance.
The experience is good! 👍

### 🎨 Blender Add-on

Our Blender add-on aims to offer a familiar UI that best feels like everything works locally, but under the hood, it communicates with a remote server where **all** simulations run, and then the results are fetched back.

This provides a unique experience where users can leverage powerful **remote** GPUs while working seamlessly in their local Blender environment. Remarkably, our Blender add-on works even on **macOS** systems 😊, unlike other CUDA-based physics simulator add-ons that require local NVIDIA GPUs.
More importantly, **you can work on a laptop without worrying about draining the battery fast**. 🔋

Follow this page [How to Install](https://st-tech.github.io/ppf-contact-solver/blender_addon/getting_started/install.html) to learn how to install the add-on. For a thorough walk through workflow, we refer to our documentation below:

<p align="center">
  <a href="https://st-tech.github.io/ppf-contact-solver"><img src="./asset/image/read-blender-addon-docs.svg" alt="Read Blender Add-on Documentation" height="35"></a>
</p>

Here are some highlights:

#### 📖 Docs Look

We maintain a [full docs site](https://st-tech.github.io/ppf-contact-solver) with workflow guides and recorded walkthroughs for the add-on:

<table>
<tr>
<td width="50%" valign="top"><img src="./docs/blender_addon/images/screenshots/documentation-dark-small.png" alt="Workflow documentation page on the docs site, showing the Blender / solver coordinate system explanation."></td>
<td width="50%" valign="top"><img src="./docs/blender_addon/images/screenshots/video-tutorials-dark-small.jpg" alt="Video Tutorials page on the docs site, showing a grid of recorded walkthroughs."></td>
</tr>
<tr>
<td valign="top">Workflow documentation page. <a href="https://st-tech.github.io/ppf-contact-solver/blender_addon/workflow/index.html">(Link)</a></td>
<td valign="top">Video tutorials page. <a href="https://st-tech.github.io/ppf-contact-solver/blender_addon/tutorial.html">(Link)</a></td>
</tr>
</table>

#### 🖼️ UI Look

Here are a couple of screenshots of the add-on running inside Blender:

<table>
<tr>
<td width="50%" valign="top"><img src="./docs/blender_addon/images/gallery/kite-ui-small.jpg" alt="The kite scene set up inside Blender using the add-on."></td>
<td width="50%" valign="top"><img src="./docs/blender_addon/images/gallery/zebra-ui-small.jpg" alt="The zebra scene set up inside Blender using the add-on."></td>
</tr>
<tr>
<td valign="top">Kite scene set up in Blender. <a href="https://zozo.box.com/s/dbtktx71fd0gb4z2trnvew7l3t8fuwxu">(full-size)</a></td>
<td valign="top">Zebra scene set up in Blender. <a href="https://zozo.box.com/s/bkt5uviyqx825os7r854xslurqpxcj2k">(full-size)</a></td>
</tr>
</table>

#### 🤖 From Natural Language to Simulation (via MCP)

We expose all of the add-on's tools through an MCP server, so any LLM (Claude, Codex, etc.) can drive the whole pipeline from a natural language prompt. Scene building, parameter tweaks, and running the simulation all happen without UI clicks. Here are two examples:

<table>
<tr>
<td width="50%" valign="top"><img src="./docs/blender_addon/images/gallery/mcp.webp" alt="Codex terminal on the left driving Blender on the right through the MCP server, building a bowl-and-spheres scene from a natural language prompt."></td>
<td width="50%" valign="top"><img src="./docs/blender_addon/images/gallery/drape-over-sphere.webp" alt="A cloth sheet draped over a sphere, produced from a single natural language prompt through the MCP server."></td>
</tr>
<tr>
<td valign="top">Codex (left) driving Blender (right) through the add-on's MCP server.</td>
<td valign="top">A prompt: drape a sheet over a sphere and make an animation video mp4 render 300 frames.</td>
</tr>
</table>

#### 🐍 From a Python Script to Simulation

You can also drive the entire pipeline from a Python script inside Blender's scripting editor. This is handy for procedural scene setup and batch variant generation. Below is a full example that drapes a sheet over a sphere:

```python
import bpy
from zozo_contact_solver import solver

# Reset any prior state.
solver.clear()

# Create a sphere (the static collider) at the origin.
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=0.5, location=(0, 0, 0))
bpy.context.object.name = "Sphere"

# Create a 2x2 sheet just above the sphere as a 64x64 grid.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=64, y_subdivisions=64, size=2, location=(0, 0, 0.6))
sheet = bpy.context.object
sheet.name = "Sheet"

# Pin the two corners on the -x edge via a vertex group.
vg = sheet.vertex_groups.new(name="Corners")
corner_indices = [
    i for i, v in enumerate(sheet.data.vertices)
    if v.co.x < -0.99 and abs(abs(v.co.y) - 1.0) < 0.01
]
vg.add(corner_indices, 1.0, "REPLACE")

# Build solver groups.
cloth = solver.create_group("Cloth", type="SHELL")
cloth.add("Sheet")
cloth.param.enable_strain_limit = True
cloth.param.strain_limit = 0.05
cloth.param.bend = 1

ball = solver.create_group("Ball", type="STATIC")
ball.add("Sphere")

# Pin the two sheet corners.
cloth.create_pin("Sheet", "Corners")

# Scene parameters.
solver.param.frame_count = 100
solver.param.step_size = 0.01
```

Here's how the script runs inside Blender [(full-size)](https://zozo.box.com/s/m86w3jyprvz5dug2pr7k73bozdbgldfl):

![python-scripting](./docs/blender_addon/images/screenshots/python-scripting.jpg)

For the full `solver.*` surface, see the [Blender Python API guide](https://st-tech.github.io/ppf-contact-solver/blender_addon/integrations/python_api.html).

### 🌐 JupyterLab

Our frontend is accessible through a browser using our built-in JupyterLab interface.
All is set up when you open it for the first time.
Results can be interactively viewed through the browser and exported as needed.
Our Python interface is designed with the following principles in mind:

- **🛠️ In-Pipeline Tri/Tet Creation**: Depending on external 3D/CAD softwares for triangulation or tetrahedralization makes dynamic resolution changes cumbersome. We provide handy `.triangulate()` and `.tetrahedralize()` calls to keep everything in-pipeline, allowing users to skip explicit mesh exports to 3D/CAD software.
- **🚫 No Mesh Data Included**: Preparing mesh data using external tools can be cumbersome. Our frontend minimizes this effort by allowing meshes to be created on the fly or downloaded when needed.
- **🔗 Method Chaining**: We adopt the method chaining style from JavaScript, making the API intuitive to understand and read smoothly.
- **📦 Single Import for Everything**: All frontend features are accessible by simply importing with `from frontend import App`.

Here's an example of draping five sheets over a sphere with two corners pinned.
We have more examples in the [examples](./examples/) directory. Please take a look! 👀

```python
# import our frontend
from frontend import App

# make an app
app = App.create("drape")

# create a square mesh resolution 128 spanning the xz plane
V, F = app.mesh.square(res=128, ex=[1, 0, 0], ey=[0, 0, 1])

# add to the asset and name it "sheet"
app.asset.add.tri("sheet", V, F)

# create an icosphere mesh radius 0.5
V, F = app.mesh.icosphere(r=0.5, subdiv_count=4)

# add to the asset and name it "sphere"
app.asset.add.tri("sphere", V, F)

# create a scene
scene = app.scene.create()

# define gap between sheets
gap = 0.01

for i in range(5):

    # add the sheet asset to the scene with an vertical offset
    obj = scene.add("sheet").at(0, gap * i, 0)

    # pick two corners
    corner = obj.grab([1, 0, -1]) + obj.grab([-1, 0, -1])

    # pin the corners
    obj.pin(corner)

    # set the strict limit on maximum strain to 5% per triangle
    obj.param.set("strain-limit", 0.05)

# add a sphere mesh at a lower position with jitter and set it static collider
scene.add("sphere").at(0, -0.5 - gap, 0).jitter().pin()

# compile the scene and report stats
scene = scene.build().report()

# preview the initial scene, shows image left
scene.preview()

# create a new session with the compiled scene
session = app.session.create(scene)

# set session params
session.param.set("frames", 100).set("dt", 0.01)

# build this session
session = session.build()

# start the simulation and live-preview the results, shows image right
session.start().preview()

# also show streaming logs
session.stream()

# or interactively view the animation sequences
session.animate()

# export all simulated frames in (sequences of ply meshes + a video)
session.export.animation()
```

<img src="./asset/image/drape-preview.webp" alt="drape">

#### 📚 Python APIs and Parameters

- Full API documentation is available on our [GitHub Pages](https://st-tech.github.io/ppf-contact-solver/jupyterlab_api/module_reference.html). The major APIs are documented using docstrings and compiled with [Sphinx](https://www.sphinx-doc.org/en/master/)
We have also included [`jupyter-lsp`](https://github.com/jupyter-lsp/jupyterlab-lsp) to provide interactive linting assistance and display docstrings as you type. See this video [(Video)](https://zozo.box.com/s/52atrfn70vn8u07iwzbyrrz5ameczo03) for an example.
The behaviors can be changed through the settings.

- A list of parameters used in `param.set(key,value)` is documented here: [(Simulation Parameters)](https://st-tech.github.io/ppf-contact-solver/jupyterlab_api/simulation_parameters.html) [(Material Parameters)](https://st-tech.github.io/ppf-contact-solver/jupyterlab_api/material_parameters.html).

> ⚠️ Please note that our Python APIs are subject to breaking changes as this repository undergoes frequent iterations. If you need APIs to be fixed, please fork.

## 🔍 Obtaining Logs

Logs for the simulation can also be queried through our Python APIs. Here's an example of how to get a list of recorded logs, fetch them, and compute the average.

```python
# get a list of log names
logs = session.get.log.names()
print(logs)
assert "time-per-frame" in logs
assert "newton-steps" in logs

# get a list of time per video frame
msec_per_video = session.get.log.numbers("time-per-frame")

# compute the average time per video frame
print("avg per frame:", sum([n for _, n in msec_per_video]) / len(msec_per_video))

# get a list of newton steps
newton_steps = session.get.log.numbers("newton-steps")

# compute the average of consumed newton steps
print("avg newton steps:", sum([n for _, n in newton_steps]) / len(newton_steps))

# Last 8 lines. Omit for everything.
print("==== log stream ====")
for line in session.get.log.stdout(n_lines=8):
    print(line)
```

Below are some representatives.
`vid_time` refers to the video time in seconds and is recorded as `float`.
`ms` refers to the consumed simulation time in milliseconds recorded as `int`.
`vid_frame` is the video frame count recorded as `int`.

| **Name** | **Description** | **Format**
|---------------|----------------|------------
| time-per-frame | Time per video frame | `list[(vid_frame,ms)]` |
| matrix-assembly | Matrix assembly time | `list[(vid_time,ms)]` |
| pcg-linsolve | Linear system solve time | `list[(vid_time,ms)]` |
| line-search | Line search time | `list[(vid_time,ms)]` |
| time-per-step | Time per step | `list[(vid_time,ms)]` |
| newton-steps | Newton iterations per step | `list[(vid_time,count)]` |
| num-contact | Contact count | `list[(vid_time,count)]` |
| max-sigma | Max stretch | `list(vid_time,float)` |

The full list of log names and their descriptions is documented here: [(GitHub Pages)](https://st-tech.github.io/ppf-contact-solver/jupyterlab_api/log_channels.html).

Note that some entries have multiple records at the same video time. This occurs because the same operation is executed multiple times within a single step during the inner Newton's iterations. For example, the linear system solve is performed at each Newton's step, so if multiple Newton's steps are executed, multiple linear system solve times appear in the record at the same video time.

If you would like to retrieve the raw log stream, you can do so by

```python
# Last 8 lines. Omit for everything.
for line in session.get.log.stdout(n_lines=8):
    print(line)
```

This will output something like:

```text
* dt: 1.000e-03
* max_sigma: 1.045e+00
* avg_sigma: 1.030e+00
------ newton step 1 ------
   ====== contact_matrix_assembly ======
   > dry_pass...0 msec
   > rebuild...7 msec
   > fillin_pass...0 msec
```

If you would like to read `stderr`, you can do so using `session.get.stderr()` (if it exists).
This returns `list[str]`.
All the log files are updated in real-time and can be fetched right after the simulation starts; you don't have to wait until it finishes.

## 🖼️ Catalogue

### 🎨 Blender Add-on Examples

These scenes are all built directly in Blender using our [add-on](#-blender-add-on). You set the geometry, constraints, and parameters from Blender's UI, and the simulation itself runs on a remote solver, or directly on your local machine if you have a modern NVIDIA GPU on Windows or Linux.

||||
|---|---|---|
|[kite.blend](https://zozo.box.com/s/j5tg9hy7nf1fdea1yg0s6holzwwus77t) [(Video)](https://zozo.box.com/s/7siwyp04s1vs48znwnt5vx1vhgodyr4h)|[crumple.blend](https://zozo.box.com/s/69ysygaqfud3bba8v3w33eqcbfn6l76u) [(Video)](https://zozo.box.com/s/ddrqqq87gpn0mekqx0yaukez87casten)|[puff.blend](https://zozo.box.com/s/mfc64djyjyunuhnmn51rm4jexxenx0si) [(Video)](https://zozo.box.com/s/8dpuoqbg80vxvxwsga36nz1633vx4k6u)|
|![](./docs/blender_addon/images/gallery/kite.webp)|![](./docs/blender_addon/images/gallery/crumple.webp)|![](./docs/blender_addon/images/gallery/puff.webp)|
|[press.blend](https://zozo.box.com/s/n1upezi7j0eufmrsief4qh252of7g1nq) [(Video)](https://zozo.box.com/s/nt8s46e6kke9poruvxmtxv5v56p2ysit)|[zebra.blend](https://zozo.box.com/s/qcos081dolarpczz8mheegvwqalxcnu2) [(Video)](https://zozo.box.com/s/rvcqynftqk27fczplafm0wgt95xmhb1k)|[curtain.blend](https://zozo.box.com/s/f8775589v2jd3nnmm7dzjrfy44xmbuhl) [(Video)](https://zozo.box.com/s/e558genjdno7jz9m0svojte5eco6q7tm)|
|![](./docs/blender_addon/images/gallery/press.webp)|![](./docs/blender_addon/images/gallery/zebra.webp)|![](./docs/blender_addon/images/gallery/curtain.webp)|

### 🌐 JupyterLab Examples

All these examples run on our Python frontend through JupyterLab. Click any notebook to see how the scene is built, or click the video link to watch the result.

|||||
|---|---|---|---|
|[woven.ipynb](./examples/woven.ipynb) [(Video)](https://zozo.box.com/s/0kgqmurmahwpvufuplbse9f83lsd7cjs)|[stack.ipynb](./examples/stack.ipynb) [(Video)](https://zozo.box.com/s/cy0pmlu5wwzmhri2qhrereqgobbv4eh6)|[trampoline.ipynb](./examples/trampoline.ipynb) [(Video)](https://zozo.box.com/s/syi5q1ywgtwrqx1f9xcjgz5m4jzagdra)|[needle.ipynb](./examples/needle.ipynb) [(Video)](https://zozo.box.com/s/zlrb6xk4gg4yjm2tspyv6u2n8vlo3dyb)|
|![](./asset/image/catalogue/woven.mp4.webp)|![](./asset/image/catalogue/stack.mp4.webp)|![](./asset/image/catalogue/trampoline.mp4.webp)|![](./asset/image/catalogue/needle.mp4.webp)|
|[cards.ipynb](./examples/cards.ipynb) [(Video)](https://zozo.box.com/s/7c114pua0107xkz4nc3bwfdzpkhgn1o9)|[codim.ipynb](./examples/codim.ipynb) [(Video)](https://zozo.box.com/s/o3en7lkzz2xhi0i9hzfdidau7pp7nco2)|[hang.ipynb](./examples/hang.ipynb) [(Video)](https://zozo.box.com/s/lcgnivq93ahvp1uccw91t13ht6qggopx)|[trapped.ipynb](./examples/trapped.ipynb) [(Video)](https://zozo.box.com/s/lnnyeqrvm86rxnwyjxhojfj0jgm5nphn)|
|![](./asset/image/catalogue/cards.mp4.webp)|![](./asset/image/catalogue/codim.mp4.webp)|![](./asset/image/catalogue/hang.mp4.webp)|![](./asset/image/catalogue/trapped.mp4.webp)|
|[domino.ipynb](./examples/domino.ipynb) [(Video)](https://zozo.box.com/s/p5ksfqja1ew3c6vntco5zq6g0kgf7xoo)|[noodle.ipynb](./examples/noodle.ipynb) [(Video)](https://zozo.box.com/s/wvsn7byr4yv80wixi5cs8mgoubco6kp4)|[drape.ipynb](./examples/drape.ipynb) [(Video)](https://zozo.box.com/s/fuisqc3wpnynd46mvdc7z2gat7mkjw07)|[five-twist.ipynb](./examples/five-twist.ipynb) [(Video)](https://zozo.box.com/s/36h8jpu5vcgc5t4xln2l68afj7izsx4h)|
|![](./asset/image/catalogue/domino.mp4.webp)|![](./asset/image/catalogue/noodle.mp4.webp)|![](./asset/image/catalogue/drape.mp4.webp)|![](./asset/image/catalogue/quintupletwist.mp4.webp)|
|[ribbon.ipynb](./examples/ribbon.ipynb) [(Video)](https://zozo.box.com/s/qxsx2it6kvd6p5ufqtmge6pd63lmhcny)|[curtain.ipynb](./examples/curtain.ipynb) [(Video)](https://zozo.box.com/s/ii938zyr4ytks5bnpbch3ystztuqq8mu)|[fishingknot.ipynb](./examples/fishingknot.ipynb) [(Video)](https://zozo.box.com/s/56dqf3fmnd0qr9pptqe8u8mw8vooh01p)|[friction.ipynb](./examples/friction.ipynb) [(Video)](https://zozo.box.com/s/15r5o7rrowwtbrsrjjpj35v8xt92ufhr)|
|![](./asset/image/catalogue/ribbon.mp4.webp)|![](./asset/image/catalogue/curtain.mp4.webp)|![](./asset/image/catalogue/fishingknot.mp4.webp)|![](./asset/image/catalogue/friction.mp4.webp)|
|[belt.ipynb](./examples/belt.ipynb) [(Video)](https://zozo.box.com/s/7ovalnjrptprcib7z0hj85lvr3yuhroz)|[fitting.ipynb](./examples/fitting.ipynb) [(Video)](https://zozo.box.com/s/pdcswk4ytzqgo98cjo91ensprgxz2umm)|[roller.ipynb](./examples/roller.ipynb) [(Video)](https://zozo.box.com/s/5156ez3b7rsvldqbgxsd7irhwp7mo2nc)|[yarn.ipynb](./examples/yarn.ipynb) [(Video)](https://zozo.box.com/s/3ly6riq5id8y9ukvbpnfnlro3qttylxw)|
|![](./asset/image/catalogue/belt.mp4.webp)|![](./asset/image/catalogue/fitting.mp4.webp)|![](./asset/image/catalogue/roller.mp4.webp)|![](./asset/image/catalogue/yarn.mp4.webp)|

### 💰 Budget Table on AWS

Below is a table summarizing the estimated costs for running our examples on a NVIDIA L4 instance `g6.2xlarge` at Amazon Web Services US regions (`us-east-1` and `us-east-2`).

- 💰 Uptime cost is approximately $1 per hour.
- ⏳ Deployment time is approximately 8 minutes ($0.13). Instance loading takes 3 minutes, and Docker pull & load takes 5 minutes.
- 🎮 The NVIDIA L4 delivers [30.3 TFLOPS for FP32](https://www.nvidia.com/en-us/data-center/l4/), offering approximately 36% of the [performance of an RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
- 🎥 Video frame rate is 60fps.

| **Example** | **Cost** | **Time** | **#Frame** | **#Vert** | **#Face** | **#Tet** | **#Rod** | **Max Strain** |
|--------------|-------|-------|-----|--------|--------|--------|---------|-----|
| trapped      | $0.37 | 22.6m | 300 | 263K   | 299K   | 885K   | ```N/A```     | ```N/A``` |
| twist        | $0.91 | 55m   | 500 | 203K   | 406K   | ```N/A```    | ```N/A```     | ```N/A``` |
| stack        | $0.60 | 36.2m | 120 | 166.7K | 327.7K | 8.8K   | ```N/A```     | 5%  |
| trampoline   | $0.74 | 44.5m | 120 | 56.8K  | 62.2K  | 158.0K | ```N/A```     | 1%  |
| needle       | $0.31 | 18.4m | 120 | 86K    | 168.9K | 8.8K   | ```N/A```     | 5%  |
| cards        | $0.29 | 17.5m | 300 | 8.7K   | 13.8K  | 1.9K   | ```N/A```     | 5%  |
| domino       | $0.12 | 4.3m  | 250 | 0.5K   | 0.8K   | ```N/A```    | ```N/A```     | ```N/A``` |
| drape        | $0.10 | 3.5m  | 100 | 81.9K  | 161.3K | ```N/A```    | ```N/A```     | 5% |
| curtain      | $0.33 | 19.6m | 300 | 64K    | 124K   | ```N/A```    | ```N/A```     | 5% |
| friction     | $0.17 | 10m   | 700 | 1.1K   | ```N/A```    | 1K     | ```N/A```     | ```N/A``` |
| hang         | $0.12 | 7.5m  | 200 | 16.3K  | 32.2K  | ```N/A```    | ```N/A```     | 1%  |
| belt         | $0.19 | 11.4m | 200 | 12.3K  | 23.3K  | ```N/A```    | ```N/A```     | 5%  |
| codim        | $0.36 | 21.6m | 240 | 122.7K | 90K    | 474.1K | 1.3K    | ```N/A``` |
| fishingknot  | $0.38 | 22.5m | 830 | 19.6K  | 36.9K  | ```N/A```    | ```N/A```     | 5%  |
| fitting      | $0.03 | 1.54m | 240 | 28.4K  | 54.9K  | ```N/A```    | ```N/A```     | 10% |
| noodle       | $0.14 | 8.45m | 240 | 116.2K | ```N/A```    | ```N/A```    | 116.2K  | ```N/A``` |
| ribbon       | $0.23 | 13.9m | 480 | 34.9K  | 52.9K  | 8.8K   | ```N/A```     | 5%  |
| woven        | $0.58 | 34.6m | 450 | 115.6K | ```N/A```    | ```N/A```    | 115.4K  | ```N/A``` |
| yarn         | $0.01 | 0.24m | 120 | 28.5K  | ```N/A```    | ```N/A```    | 28.5K   | ```N/A``` |
| roller       | $0.03 | 2.08m | 240 | 21.4K  | 22.2K  | 61.0K  | ```N/A```     | ```N/A``` |

#### 🏗️ Large Scale Examples

Large scale examples are run on a [vast.ai](https://vast.ai) instance with an RTX 4090.
These examples are not included in GitHub Action tests since they can take days to finish.

| | | |
|---|---|---|
| [large-twist.ipynb](./examples/large-twist.ipynb) [(Video)](https://zozo.box.com/s/awtqwo572t7ixuy04va478zn6f69djlr) | [large-five-twist.ipynb](./examples/large-five-twist.ipynb) [(Video)](https://zozo.box.com/s/v62q7cbfnpl3hufwwy2nmewyes2w1iw6) | [large-woven.ipynb](./examples/large-woven.ipynb) [(Video)](https://zozo.box.com/s/kc81msjfo4yw9eozn8i8bean0gbph0pj) |
| ![twist](./asset/image/large-scale/twist.jpg) | ![five-twist](./asset/image/large-scale/five-twist.jpg) | ![woven](./asset/image/large-scale/woven.jpg) |

| Example | Commit | #Vert | #Face | #Rod | #Contact | #Frame | Time/Frame |
|---|---|---|---|---|---|---|---|
| large-twist | [cbafbd2](https://github.com/st-tech/ppf-contact-solver/tree/cbafbd2197fc7f28673386dfaf1e8d8a1be49937) | 3.2M | 6.4M | ```N/A``` | 56.7M | 2,000 | 46.4s |
| large-five-twist | [6ab6984](https://github.com/st-tech/ppf-contact-solver/commit/6ab6984d95f67673f1ebfdc996b0320123d88bed) | 8.2M | 16.4M | ```N/A``` | 184.1M | 2,413 | 144.5s |
| large-woven | [4c07b83](https://github.com/st-tech/ppf-contact-solver/commit/4c07b834b299e49bb08797940e9f0869789301b8) | 2.7M | ```N/A``` | 2.7M | 8.9M | 946 | 436.8s |

📝 Large scale examples take a very long time, and it's easy to lose connection or close the browser.
Our frontend lets you close and reopen it at your convenience. Just recover your session after you reconnect.
Here's an example cell how to recover:

```python
# In case you shutdown the server (or kernel) and still want
# to restart, do this.
# Do not run other cells used to create this scene.
# You can also recover this way if you closed the browser.
# Just directly run this in a new cell or in a new notebook.

from frontend import App

# recover the session
session = App.recover("app-name")

# resume if not currently running
if not App.busy():
    session.resume()

# preview the current state
session.preview()

# stream the logs
session.stream()
```

## 🚀 GitHub Actions

We implemented GitHub Actions that test all of our examples except for large scale ones, which take from days to weeks to finish.
We perform explicit intersection checks at the end of each step, which raises an error if an intersection is detected.
**This ensures that all steps are confirmed to be penetration-free if tests pass.**
The runner types are described as follows:

### [![Getting Started](https://github.com/st-tech/ppf-contact-solver/actions/workflows/getting-started.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/getting-started.yml)

The tested runner of this action is the Ubuntu NVIDIA GPU-Optimized Image for AI and HPC with an NVIDIA Tesla T4 (16 GB VRAM) with Driver version ``570.133.20``.
This is not a self-hosted runner, meaning that each time the runner launches, all environments are fresh. 🌱

### [![All Examples](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once.yml) [![All Examples (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/run-all-once-win.yml)

We use the GitHub-hosted runner, but the actual simulation runs on a `g6e.2xlarge` AWS instance.
Since we start with a fresh instance, the environment is clean every time.
We take advantage of the ability to deploy on the cloud; this action is performed in parallel, which reduces the total action time.

### [![Blender CI](https://github.com/st-tech/ppf-contact-solver/actions/workflows/blender.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/blender.yml)

This action exercises our [Blender add-on](#-blender-add-on) on free GitHub-hosted Linux and macOS runners in parallel. Blender 5.1.1 is installed from the official Blender Foundation mirror, the Rust solver is built in CPU-emulated mode (no CUDA required), and the add-on is installed as a Blender 5 extension. A headless test rig then runs the full scenario registry covering add-on UI flows.

### 📦 Action Artifacts

We generate zipped action artifacts for each run. These artifacts include:

- **📝 Logs**: Detailed logs of the simulation runs.
- **📊 Metrics**: Performance metrics and statistics.
- **📹 Videos**: Simulated animations.

Please note that these artifacts will be deleted after a month.

### ⚔️ Ten Consecutive Runs

We know that you can't judge the reliability of contact resolution by simply watching a single success video example.
To ensure greater transparency, we implemented GitHub Actions to run many of our examples via automated GitHub Actions, not just once, but **10 times in a row** for both Docker and Windows.
This means that **a single failure out of 10 tests is considered a failure of the entire test suite!**
Also, we apply small jitters to the position of objects in the scene, so **at each run, the scene is slightly different.**

##### 🪟 Windows Native

[![drape.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/drape-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/drape-win.yml)
[![cards.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/cards-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/cards-win.yml)
[![curtain.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/curtain-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/curtain-win.yml)
[![friction.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/friction-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/friction-win.yml)
[![hang.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/hang-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/hang-win.yml)
[![needle.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/needle-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/needle-win.yml)
[![stack.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/stack-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/stack-win.yml)
[![trampoline.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trampoline-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trampoline-win.yml)
[![trapped.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trapped-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trapped-win.yml)
[![twist.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/twist-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/twist-win.yml)
[![five-twist.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/five-twist-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/five-twist-win.yml)
[![domino.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/domino-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/domino-win.yml)
[![belt.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/belt-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/belt-win.yml)
[![codim.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/codim-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/codim-win.yml)
[![fishingknot.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fishingknot-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fishingknot-win.yml)
[![fitting.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fitting-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fitting-win.yml)
[![noodle.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/noodle-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/noodle-win.yml)
[![ribbon.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/ribbon-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/ribbon-win.yml)
[![woven.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/woven-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/woven-win.yml)
[![yarn.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/yarn-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/yarn-win.yml)
[![roller.ipynb (Windows Native)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/roller-win.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/roller-win.yml)

##### 🐧 Linux

[![drape.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/drape.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/drape.yml)
[![cards.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/cards.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/cards.yml)
[![curtain.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/curtain.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/curtain.yml)
[![friction.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/friction.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/friction.yml)
[![hang.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/hang.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/hang.yml)
[![needle.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/needle.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/needle.yml)
[![stack.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/stack.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/stack.yml)
[![trampoline.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trampoline.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trampoline.yml)
[![trapped.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trapped.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/trapped.yml)
[![twist.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/twist.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/twist.yml)
[![five-twist.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/five-twist.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/five-twist.yml)
[![domino.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/domino.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/domino.yml)
[![belt.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/belt.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/belt.yml)
[![codim.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/codim.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/codim.yml)
[![fishingknot.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fishingknot.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fishingknot.yml)
[![fitting.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fitting.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/fitting.yml)
[![noodle.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/noodle.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/noodle.yml)
[![ribbon.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/ribbon.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/ribbon.yml)
[![woven.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/woven.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/woven.yml)
[![yarn.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/yarn.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/yarn.yml)
[![roller.ipynb](https://github.com/st-tech/ppf-contact-solver/actions/workflows/roller.yml/badge.svg)](https://github.com/st-tech/ppf-contact-solver/actions/workflows/roller.yml)

## 📡 Deploying on Cloud Services

Running our solver on the cloud has a few practical perks:

- **💰 Pay only when you use it**: Spin up an instance, run your experiment, and delete it when you're done. You pay for hours, not for a GPU sitting on your desk.
- **📈 Scale on demand**: If you have a deadline, just launch multiple instances and run experiments in parallel. No waiting in a queue.
- **🤝 Share with collaborators**: Send the JupyterLab link to a remote teammate and they can watch the simulation right alongside you.
- **🔒 Cloud-grade security**: You inherit whatever security the provider gives you, which is usually a lot more than you'd set up yourself.
- **🐛 Easier bug repro**: When users and developers are on the same kernel and driver, reproducing a bug is far easier than juggling different local setups.
- **🛠️ No hardware to babysit**: No drivers to update, no fans to clean, no GPU to replace when it dies.

Below, we describe how to deploy our solver on major cloud services. These instructions are up to date as of late 2024 and are subject to change.

> ⚠️ For all the services below, don't forget to delete the instance after use, or you'll be charged for nothing. 💸

### 📦 Deploying on [vast.ai](https://vast.ai)

- Select our template [(Link)](https://cloud.vast.ai/?creator_id=85288&name=ppf-contact-solver).
- Create an instance and click `Open` button.

> ⚠️ `Open` button URL is public (not secure); only for testing purposes and should not be used for production use. For better security, duplicate the template and close the port, then use SSH port forwarding instead.

### 📦 Deploying on [RunPod](https://runpod.io)

- Follow this link [(Link)](https://runpod.io/console/deploy?template=we8ta2hy86&ref=bhy3csxy) and deploy an instance using our template.
- Click `Connect` button and open the `HTTP Services` link.

> ⚠️ `HTTP Services` URL is public (not secure); only for testing purposes and should not be used for production use. For better security, duplicate the template and close the port, then use SSH port forwarding instead.

### 📦 Deploying on [Scaleway](https://www.scaleway.com/en/)

- Set zone to `fr-par-2`
- Select type `L4-1-24G` or `GPU-3070-S`
- Choose `Ubuntu Jammy GPU OS 12`
- *Do not skip* the Docker container creation in the installation process; it is required.
- This setup costs approximately €0.76 per hour.
- CLI instructions are described in [(Markdown)](./articles/cloud.md#-scaleway).

### 📦 Deploying on [Amazon Web Services](https://aws.amazon.com/en/)

- Amazon Machine Image (AMI): `Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)`
- Instance Type: `g6.2xlarge` (Recommended)
- This setup costs around $1 per hour.
- *Do not skip* the Docker container creation in the installation process; it is required.

### 📦 Deploying on [Google Compute Engine](https://cloud.google.com/products/compute)

- Select `GPUs`. We recommend the GPU type `NVIDIA L4` because it's affordable and accessible, as it does not require a high quota. You may select `T4` instead for testing purposes.
- Do **not** check `Enable Virtual Workstation (NVIDIA GRID)`.
- We recommend the machine type `g2-standard-8`.
- Choose the OS type `Deep Learning VM with CUDA 12.4 M129` and set the disk size to `50GB`.
- As of late 2024, this configuration costs approximately $0.86 per hour in `us-central1 (Iowa)` and $1.00 per hour in `asia-east1 (Taiwan)`.
- Port number `8080` is reserved by the OS image. Set `$MY_WEB_PORT` to `8888`. When connecting via `gcloud`, use the following format:  `gcloud compute ssh --zone "xxxx" "instance-name" -- -L 8080:localhost:8888`.
- *Do not skip* the Docker container creation in the installation process; it is required.
- CLI instructions are described in [(Markdown)](./articles/cloud.md#-google-compute-engine).

## 🧩 Community's Blender Add-ons

Alongside our [official Blender add-on](#-blender-add-on), the following community add-ons are also available:

- [AndoSim](https://github.com/Slaymish/AndoSim)
- [ArzteZ-PPF-solver](https://github.com/tavcitavci-sys-tavci-ui/ArzteZ-PPF-solver)

## 📬 Contributing

We appreciate your interest in opening pull requests, but we are not ready to accept external contributions because doing so involves resolving copyright and licensing matters with [ZOZO, Inc.](https://corp.zozo.com/en/)
For the time being, please open issues for bug reports.
If you wish to extend the codebase, please fork the repository and work on it.
Thank you!

## 🙏 Acknowledgements

The author thanks [ZOZO, Inc.](https://corp.zozo.com/en/) for permitting the release of the code and the team members for assisting with the internal paperwork for this project.
This repository is owned by [ZOZO, Inc.](https://corp.zozo.com/en/)
