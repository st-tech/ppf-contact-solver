# 📡 Deploying on Cloud Services

Running our solver on the cloud has a few practical advantages:

- **💰 Pay only when you use it**: Spin up an instance, run your experiment, and delete it when you're done. You pay for hours, not for a GPU sitting on your desk.
- **📈 Scale on demand**: If you have a deadline, just launch multiple instances and run experiments in parallel. No waiting in a queue.
- **🤝 Share with collaborators**: Send the JupyterLab link to a remote teammate and they can watch the simulation right alongside you.
- **🔒 Cloud-grade security**: You inherit whatever security the provider gives you, which is usually a lot more than you'd set up yourself.
- **🐛 Reproducible environments**: Users and developers share the same kernel and driver, making bug reproduction more reliable than across heterogeneous local setups.
- **🛠️ No hardware maintenance**: No driver updates, no thermal management, and no replacement costs when hardware fails.

Below, we describe how to deploy our solver on major cloud services. These instructions are up to date as of late 2024 and are subject to change.

> ⚠️ For all the services below, don't forget to delete the instance after use, or you'll be charged for nothing. 💸

## 📦 Deploying on [vast.ai](https://vast.ai)

- Select our template [(Link)](https://cloud.vast.ai?ref_id=85288&template_id=29158e5c91e1b4b9543b6a027a837979).
- For a step-by-step video tutorial, see ["Setting Up vast.ai"](https://st-tech.github.io/ppf-contact-solver/blender_addon/tutorial.html).
- Create an instance and connect via SSH with port forwarding, e.g. `ssh -L 8080:localhost:8080 root@<host> -p <port>`, then open `http://localhost:8080` in your browser.

## 📦 Deploying on [Scaleway](https://www.scaleway.com/en/)

- Set zone to `fr-par-2`
- Select type `L4-1-24G` or `GPU-3070-S`
- Choose `Ubuntu Jammy GPU OS 12`
- *Do not skip* the Docker container creation in the installation process; it is required.
- This setup costs approximately €0.76 per hour.
- CLI instructions are described in [(Markdown)](./cloud.md#-scaleway).

## 📦 Deploying on [Amazon Web Services](https://aws.amazon.com/en/)

- Amazon Machine Image (AMI): `Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)`
- Instance Type: `g6.2xlarge` (Recommended)
- This setup costs around $1 per hour.
- *Do not skip* the Docker container creation in the installation process; it is required.

## 📦 Deploying on [Google Compute Engine](https://cloud.google.com/products/compute)

- Select `GPUs`. We recommend the GPU type `NVIDIA L4` because it's affordable and accessible, as it does not require a high quota. You may select `T4` instead for testing purposes.
- Do **not** check `Enable Virtual Workstation (NVIDIA GRID)`.
- We recommend the machine type `g2-standard-8`.
- Choose a Deep Learning VM OS image from the family `common-cu129-ubuntu-2404-nvidia-580` (NVIDIA driver 580) and set the disk size to `50GB`.
- As of late 2024, this configuration costs approximately $0.86 per hour in `us-central1 (Iowa)` and $1.00 per hour in `asia-east1 (Taiwan)`.
- Port number `8080` is reserved by the OS image. Set `$MY_WEB_PORT` to `8888`. When connecting via `gcloud`, use the following format:  `gcloud compute ssh --zone "xxxx" "instance-name" -- -L 8080:localhost:8888`.
- *Do not skip* the Docker container creation in the installation process; it is required.
- CLI instructions are described in [(Markdown)](./cloud.md#-google-compute-engine).
