## ⭐ Cleaner Installation

> [!NOTE]
> If you wish to install our solver on a headless remote machine, SSH into the server with port forwarding using the following command:
>
> ```bash
> MY_WEB_PORT=8080
> ssh -L 8080:localhost:$MY_WEB_PORT user@remote_server_address
> ```
>
> This port forwarding will be used to access the frontend afterward.
> The second port number must match `$MY_WEB_PORT` below.
>
> In this case, `$MY_WEB_PORT` refers to a port on the host machine, not on your local computer. In this example, you'll be connecting to the frontend via <http://localhost:8080>, regardless of the value of `$MY_WEB_PORT`.

### 🎥 Installation Videos

We provide uninterrupted recorded installation videos (🪟 Windows [(Video)](https://zozo.box.com/s/cepynbieg3vdvv3q3ix1ybn2g0ix8531), 🐧 Linux [(Video)](https://zozo.box.com/s/w4doo2hjirds3qhlzk62dzu6csxatpyo), ☁ [vast.ai](https://vast.ai) [(Video)](https://zozo.box.com/s/nigd94wnjy6jkxo7e0zlbfnzausyf22b), and ☁️ [Google Cloud](https://cloud.google.com/products/compute) [(Video)](https://zozo.box.com/s/y9wvc901yud55waxaesl62djrl32o498))
to reduce stress during the installation process. We encourage you to check them out to get a sense of how things go and how long each step takes.

### 🪟 Windows

Create a container by running the following Docker command in PowerShell:

```
$MY_WEB_PORT = 8080  # Web port number on the host
$MY_TIME_ZONE = "Asia/Tokyo"  # Your time zone
$MY_CONTAINER_NAME = "ppf-contact-solver"  # Container name

docker run -it `
    --gpus all `
    -p ${MY_WEB_PORT}:${MY_WEB_PORT} `
    -e WEB_PORT=${MY_WEB_PORT} `
    -e TERM `
    -e TZ=$MY_TIME_ZONE `
    -e LANG=en_US.UTF-8 `
    --hostname ppf-dev `
    --name $MY_CONTAINER_NAME `
    -e NVIDIA_DRIVER_CAPABILITIES="graphics,compute,utility" `
    nvidia/cuda:12.8.0-devel-ubuntu24.04
```

### 🐧 Linux

Create a container by running the following Docker command in bash/zsh:

```
MY_WEB_PORT=8080  # Web port number on the host
MY_TIME_ZONE=Asia/Tokyo  # Your time zone
MY_CONTAINER_NAME=ppf-contact-solver  # Container name

docker run -it \
    --gpus all \
    -p $MY_WEB_PORT:$MY_WEB_PORT \
    -e WEB_PORT=$MY_WEB_PORT \
    -e TERM -e TZ=$MY_TIME_ZONE \
    -e LANG=en_US.UTF-8 \
    --hostname ppf-dev \
    --name $MY_CONTAINER_NAME -e \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    nvidia/cuda:12.8.0-devel-ubuntu24.04
```

### 🪟🐧 Both Systems

At the end of the line, you should see:

```
root@ppf-dev:/#
```

From here on, all commands will happen in the container, not on your host.
Next, we'll make sure that a NVIDIA driver is visible from the Docker container. Try this

```
nvidia-smi
```

> [!NOTE]
> If an error occurs, ensure that `nvidia-smi` is working on your host. For Linux users, make sure the NVIDIA Container Toolkit is properly installed. If the issue persists, try running `sudo service docker restart` on your host to resolve it.

Please confirm that your GPU is listed here.
Now let's get the installation started.
No worries; all the commands below only disturb things in the container, so your host environment stays clean.
First, install following packages

```
apt update
apt install -y git python3
```

Next, clone our repository

```
git clone https://github.com/st-tech/ppf-contact-solver.git
```

Move into the ```ppf-contact-solver``` and let ```warmup.py``` do all the rest:

> [!NOTE]
> If you’re suspicious, you can look around ```warmup.py``` before you proceed. Run `less warmup.py`, scroll all the way to the bottom, and hit `q` to quit.

```
cd ppf-contact-solver
python3 warmup.py
```

Now we're set. Let's kick in the compilation!

```
source "$HOME/.cargo/env"
cargo build --release
```

Be patient; this takes some time. If the last line says

```
Finished `release` profile [optimized] target(s) in ...
```

We're done! 🎉 Start our frontend by

```
python3 warmup.py jupyter
```

and now you can access our JupyterLab frontend from <http://localhost:8080> on your browser.

> [!NOTE]
> The port number `8080` corresponds to the value set for `$MY_WEB_PORT` when the host machine and the local machine are the same. If you're connected to the host computer via SSH port forwarding, the first port option in the command (e.g., `xxxx` in `-L xxxx:localhost:$MY_WEB_PORT`) is the port number.

### 🧹 Cleaning Up

To remove all traces, simply stop the container and delete it.
Be aware that all simulation data will be also lost. Back up any important data if needed.

```
docker stop $MY_CONTAINER_NAME
docker rm $MY_CONTAINER_NAME
```

> [!NOTE]
> If you wish to completely wipe what we’ve done here, you may also need to purge the Docker image by:
>
> ```
> docker rmi $(docker images | grep 'nvidia/cuda' | grep '12.8.0-devel-ubuntu24.04' | awk '{print $3}')
> ```
>
> but don't do this if you still need it.

