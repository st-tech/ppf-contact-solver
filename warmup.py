# File: warmup.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import subprocess
import sys
import os


def create_clang_config():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    eigsys_dir = os.path.join(script_dir, "eigsys")
    clang_format = [
        "BasedOnStyle: LLVM",
        "IndentWidth: 4",
    ]
    clangd = [
        "CompileFlags:",
        "  Add:",
        '    - "-I/usr/include/eigen3"',
        '    - "-I/usr/local/cuda/include"',
        f'    - "-I{eigsys_dir}"',
        '    - "--no-cuda-version-check"',
        "Diagnostics:",
        "  UnusedIncludes: None",
        "  ClangTidy:",
        "    Remove: misc-definitions-in-headers",
    ]
    name_1, name_2 = ".clang-format", ".clangd"
    if not os.path.exists(name_1):
        with open(name_1, "w") as f:
            f.write("\n".join(clang_format))
            f.write("\n")
    if not os.path.exists(name_2):
        with open(name_2, "w") as f:
            f.write("\n".join(clangd))
            f.write("\n")


def list_packages():
    packages = [
        "build-essential",
        "clang",
        "clangd",
        "wget",
        "unzip",
        "cmake",
        "python3-venv",
        "xorg-dev",
        "libgl1-mesa-dev",
        "libglu1-mesa-dev",
        "libc++-dev",
        "libeigen3-dev",
    ]
    return packages


def python_packages():
    python_packages = [
        "click",
        "psutil",
        "numpy",
        "libigl",
        "plyfile",
        "requests",
        "trimesh",
        "pywavefront",
        "matplotlib",
        "tqdm",
        "pythreejs",
        "ipywidgets",
        "open3d",
        "gpytoolbox",
        "jupyterlab",
        "tabulate",
        "tetgen",
        "triangle",
    ]
    return python_packages


def install_lazygit():
    if not os.path.exists("/usr/local/bin/lazygit"):
        cmd = 'curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po \'"tag_name": "v\\K[^"]*\''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        latest_version = result.stdout.strip().replace("v", "")
        print(f"Latest version of lazygit: {latest_version}")
        url = f"https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_{latest_version}_Linux_x86_64.tar.gz"
        subprocess.run(["curl", "-Lo", "lazygit.tar.gz", url], cwd="/tmp")
        subprocess.run(["tar", "xf", "lazygit.tar.gz"], cwd="/tmp")
        subprocess.run(["install", "lazygit", "/usr/local/bin"], cwd="/tmp")


def install_nvim():
    run(
        "curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz"
    )
    run("tar -C /opt -xzf nvim-linux64.tar.gz")
    run("ln -s /opt/nvim-linux64/bin/nvim /usr/bin/nvim")
    run("curl -fsSL https://deb.nodesource.com/setup_21.x | bash -")
    run("apt install -y nodejs")
    run("curl https://www.npmjs.com/install.sh | sh")
    run("apt install -y fzf fd-find bat")
    run(
        "curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher"
    )
    run("ln -s $(which fdfind) /usr/bin/fd")
    run("ln -s $(which batcat) /usr/bin/bat")
    run("fisher install PatrickF1/fzf.fish@v7.0")


def install_fish():
    run("apt-add-repository ppa:fish-shell/release-3")
    run("apt update")
    run("apt install -y fish")
    run("chsh -s /usr/bin/fish")
    run("fish -c exit")
    run("echo 'fish_add_path $HOME/.cargo/bin' >> ~/.config/fish/config.fish")
    run("echo 'fish_add_path /usr/local/cuda/bin' >> ~/.config/fish/config.fish")


def install_oh_my_zsh():
    run("apt install -y zsh")
    run(
        'sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"'
    )
    run("zsh -c exit")
    run("echo 'export PATH=$HOME/.cargo/bin:$PATH' >> ~/.zshrc")


def run(command, cwd="/tmp"):
    if not os.path.exists("warmup.py"):
        print("Please run this script in the same directory as warmup.py")
        sys.exit(1)
    subprocess.run(command, shell=True, cwd=cwd)


def setup():
    run("apt update")
    run("apt install -y locales-all")
    run("DEBIAN_FRONTEND=noninteractive apt install -y " + " ".join(list_packages()))
    run("pip3 install " + " ".join(python_packages()))
    run("git clone https://github.com/skoch9/meshplot /tmp/meshplot")
    run("pip3 install /tmp/meshplot")
    run("curl https://sh.rustup.rs -sSf | sh -s -- -y")


def set_tmux():
    run("apt install -y tmux")
    tmux_config_commands = [
        "set-option -g prefix C-t",
        "set-option -g status off",
        "set-option -sg escape-time 10",
        'set-option -g default-terminal "screen-256color"',
        "set-option -g focus-events on",
        "unbind-key C-b",
        "bind-key C-t send-prefix",
        "bind h select-pane -L",
        "bind j select-pane -D",
        "bind k select-pane -U",
        "bind l select-pane -R",
    ]
    with open(os.path.expanduser("~/.tmux.conf"), "w") as f:
        for command in tmux_config_commands:
            f.write(command + "\n")


def set_time():
    run("apt-get install ntp")


def start_jupyter():
    import click
    import psutil
    for process in psutil.process_iter():
        if "jupyter" in process.name():
            print("Jupyter is already running")
            if click.confirm("Do you want to restart?", default=True):
                psutil.Process(process.pid).terminate()
            else:
                return
    script_dir = os.path.dirname(os.path.realpath(__file__))
    examples_dir = os.path.join(script_dir, "examples")

    override_file = "/usr/local/share/jupyter/lab/settings/overrides.json"
    if not os.path.exists(override_file):
        os.makedirs(os.path.dirname(override_file), exist_ok=True)
        with open(override_file, "w") as f:
            lines = """{
  "@jupyterlab/apputils-extension:themes": {
    "theme": "JupyterLab Dark"
  }
}
"""
            f.write(lines)

    command = "jupyter-lab --no-browser --port=8080 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password=''"
    run(command, cwd=examples_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "nvim":
            create_clang_config()
            install_nvim()
            install_lazygit()
        elif mode == "fish":
            install_fish()
        elif mode == "ohmyzsh":
            install_oh_my_zsh()
        elif mode == "tmux":
            set_tmux()
        elif mode == "clangd":
            create_clang_config()
        elif mode == "time":
            set_time()
        elif mode == "jupyter":
            start_jupyter()
        elif mode == "all":
            create_clang_config()
            install_nvim()
            install_fish()
            set_tmux()
            install_lazygit()
    else:
        setup()
