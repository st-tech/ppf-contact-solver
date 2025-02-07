# File: warmup.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import subprocess
import sys
import os


def run(command, cwd="/tmp"):
    if not os.path.exists("warmup.py"):
        print("Please run this script in the same directory as warmup.py")
        sys.exit(1)
    subprocess.run(command, shell=True, cwd=cwd)


def create_clang_config():
    print("setting up clang config")
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


def create_vscode_ext_recommend():
    print("setting up vscode extension recommendation")
    text = """{
    "recommendations": [
        "llvm-vs-code-extensions.vscode-clangd"
    ]
}"""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(script_dir, ".vscode", "extensions.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def list_packages():
    packages = [
        "curl",
        "python3-pip",
        "build-essential",
        "clang",
        "clangd",
        "wget",
        "zip",
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
        "numpy",
        "libigl",
        "plyfile",
        "requests",
        "trimesh",
        "pyrender",
        "pywavefront",
        "matplotlib",
        "tqdm",
        "pythreejs",
        "ipywidgets",
        "open3d",
        "gpytoolbox",
        "tabulate",
        "tetgen",
        "triangle",
        "ruff",
        "black",
        "isort",
        "jupyterlab",
        "jupyterlab-lsp",
        "python-lsp-server",
        "python-lsp-ruff",
        "jupyterlab-code-formatter",
    ]
    return python_packages


def install_lazygit():
    if not os.path.exists("/usr/local/bin/lazygit"):
        print("installing lazygit")
        cmd = 'curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po \'"tag_name": "v\\K[^"]*\''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        latest_version = result.stdout.strip().replace("v", "")
        print(f"Latest version of lazygit: {latest_version}")
        url = f"https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_{latest_version}_Linux_x86_64.tar.gz"
        subprocess.run(["curl", "-Lo", "lazygit.tar.gz", url], cwd="/tmp")
        subprocess.run(["tar", "xf", "lazygit.tar.gz"], cwd="/tmp")
        subprocess.run(["install", "lazygit", "/usr/local/bin"], cwd="/tmp")


def install_nvim():
    print("installing nvim")
    run(
        "curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz"
    )
    run("tar -C /opt -xzf nvim-linux-x86_64.tar.gz")
    run("ln -s /opt/nvim-linux-x86_64/bin/nvim /usr/bin/nvim")
    run("apt install -y fzf fd-find bat")
    run("/root/.cargo/bin/rustup component add rust-analyzer")
    run("ln -s $(which fdfind) /usr/bin/fd")
    run("ln -s $(which batcat) /usr/bin/bat")


def install_lazyvim():
    print("installing lazyvim")
    run("git clone https://github.com/LazyVim/starter ~/.config/nvim")
    run("rm -rf ~/.config/nvim/.git")


def install_fish():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print("installing fish")
    run("apt-add-repository ppa:fish-shell/release-3")
    run("apt update")
    run("apt install -y fish")
    run("chsh -s /usr/bin/fish")
    run("fish -c exit")
    run("echo 'fish_add_path $HOME/.cargo/bin' >> ~/.config/fish/config.fish")
    run("echo 'fish_add_path /usr/local/cuda/bin' >> ~/.config/fish/config.fish")
    run(
        f"echo 'set -x PYTHONPATH $PYTHONPATH {script_dir}' >> ~/.config/fish/config.fish"
    )


def install_oh_my_zsh():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print("installing oh-my-zsh")
    run("apt install -y zsh")
    run(
        'sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"',
        cwd=script_dir,
    )
    run("zsh -c exit")
    run("echo 'export PATH=$HOME/.cargo/bin:$PATH' >> ~/.zshrc")
    run("echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.zshrc")
    run(f"echo 'export PYTHONPATH={script_dir}:$PYTHONPATH' >> ~/.zshrc")


def install_meshplot():
    run("git clone https://github.com/skoch9/meshplot /tmp/meshplot")
    run("pip3 install --ignore-installed /tmp/meshplot")


def install_sdf():
    run("git clone https://github.com/fogleman/sdf.git /tmp/sdf")
    run("pip3 install /tmp/sdf")


def install_mesa():
    run("apt update")
    run(
        "wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb -O /tmp/mesa_18.3.3-0.deb"
    )
    run("dpkg -i /tmp/mesa_18.3.3-0.deb || true")
    run("apt install -y -f")


def setup():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    run("apt update")
    run("apt install -y locales-all")
    run("DEBIAN_FRONTEND=noninteractive apt install -y " + " ".join(list_packages()))
    run("pip3 install --ignore-installed " + " ".join(python_packages()))
    install_meshplot()
    install_sdf()
    run("curl -fsSL https://deb.nodesource.com/setup_21.x | bash -")
    run("apt install -y nodejs")
    run("curl https://www.npmjs.com/install.sh | sh")
    run("curl https://sh.rustup.rs -sSf | sh -s -- -y")
    run(f"echo 'export PYTHONPATH={script_dir}:$PYTHONPATH' >> ~/.bashrc")
    install_mesa()


def set_tmux():
    print("installing tmux")
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
    run("pkill jupyter-lab")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    examples_dir = os.path.join(script_dir, "examples")

    lsp_symlink = os.path.join(examples_dir, ".lsp_symlink")
    if not os.path.exists(lsp_symlink):
        run(f"ln -s / {lsp_symlink}")

    config_path = os.path.expanduser("~/.ipython/profile_default/ipython_config.py")
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write("c = get_config()\n")
            f.write("c.Completer.use_jedi = False")

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

    try:
        command = "jupyter-lab -y --no-browser --port=8080 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"
        env = os.environ.copy()
        key = "PYTHONPATH"
        if key in env:
            if script_dir not in env[key]:
                env[key] += f":{script_dir}"
        else:
            env[key] = script_dir
        subprocess.run(command, shell=True, cwd=examples_dir, env=env)
    except KeyboardInterrupt:
        print("jupyterlab shutdown")


def build_docs():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    run("sphinx-build -b html ./ ./_build", cwd=os.path.join(script_dir, "docs"))


def make_docs():
    from frontend import App

    with open(os.path.join("docs", "parameters.rst"), "w") as file:
        file.write(export_param_sphinx(App.get_default_param()))
    with open(os.path.join("docs", "logs.rst"), "w") as file:
        file.write(export_log_sphinx())
    print("Sphinx .rst files has been exported.")


def export_param_sphinx(param):
    rst_content = []

    title = "parameters"
    rst_content.append(f"{title}\n")
    rst_content.append("=" * len(title) + "\n\n")

    for name, entry in param.items():
        doc = entry["doc"]
        if doc["list"]:
            rst_content.append(f"{name}\n")
            rst_content.append("-" * len(name) + "\n\n")

            rst_content.append(".. list-table::\n\n")
            rst_content.append("   * - Default Value\n")

            var_type = entry["type"]
            type_dict = {
                "bool": "bool",
                "u8": "int",
                "u32": "int",
                "i32": "int",
                "f32": "float",
                "f64": "float",
                "String": "str",
            }
            type_str = type_dict[var_type]
            if type_str == "bool":
                value_str = "False"
            else:
                value_str = entry["value"]
                if type_str == "str":
                    value_str = f'"{value_str}"'
            rst_content.append(f"     - {value_str} ({type_str})\n")

            for key, value in doc.items():
                if key != "list" and value:
                    rst_content.append(f"   * - {key}\n")
                    rst_content.append(f"     - {value}\n")

            rst_content.append("\n")

    return "".join(rst_content)


def export_log_sphinx():
    from frontend import CppRustDocStringParser

    script_dir = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.join(script_dir, "src")
    result = CppRustDocStringParser.get_logging_docstrings(src_dir)

    rst_content = []

    title = "log lookup names"
    rst_content.append(f"{title}\n")
    rst_content.append("=" * len(title) + "\n\n")

    for name, doc in result.items():
        rst_content.append(f"{name}\n")
        rst_content.append("-" * len(name) + "\n\n")

        rst_content.append(".. list-table::\n\n")

        for key, value in doc.items():
            if key != "filename" and value:
                rst_content.append(f"   * - {key}\n")
                rst_content.append(f"     - {value}\n")

        rst_content.append("\n")

    return "".join(rst_content)


if __name__ == "__main__":
    if not os.path.exists(os.path.expanduser("~/.config")):
        os.makedirs(os.path.expanduser("~/.config"))
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "nvim":
            install_nvim()
        elif mode == "lazyvim":
            install_lazyvim()
        elif mode == "lazygit":
            install_lazygit()
        elif mode == "fish":
            install_fish()
        elif mode == "ohmyzsh":
            install_oh_my_zsh()
        elif mode == "tmux":
            set_tmux()
        elif mode == "clangd":
            create_clang_config()
        elif mode == "vscode":
            create_vscode_ext_recommend()
        elif mode == "time":
            set_time()
        elif mode == "jupyter":
            start_jupyter()
        elif mode == "docs-prepare":
            run("pip3 install " + " ".join(python_packages()))
            run("pip3 install sphinx sphinxawesome-theme sphinx_autobuild")
            install_meshplot()
            install_sdf()
        elif mode == "docs-build":
            make_docs()
            build_docs()
        elif mode == "all":
            create_clang_config()
            install_nvim()
            install_fish()
            set_tmux()
            install_lazygit()
            install_lazyvim()
    else:
        setup()
