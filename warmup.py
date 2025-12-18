# File: warmup.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import platform
import shutil
import subprocess
import sys


def run(command, cwd="/tmp", use_sudo=False, check=True):
    if not os.path.exists("warmup.py"):
        print("Please run this script in the same directory as warmup.py")
        sys.exit(1)
    if use_sudo and shutil.which("sudo"):
        command = f"sudo {command}"
    result = subprocess.run(command, shell=True, cwd=cwd, check=check)
    return result


def get_venv_path():
    venv_dir = os.path.expanduser("~/.local/share/ppf-cts")
    os.makedirs(venv_dir, exist_ok=True)
    return os.path.join(venv_dir, "venv")


def create_venv():
    venv_path = get_venv_path()
    if not os.path.exists(venv_path):
        print(f"Creating virtual environment at {venv_path}")
        result = subprocess.run([sys.executable, "-m", "venv", venv_path])
        if result.returncode != 0:
            print("Failed to create virtual environment")
            print("Installing python3-venv...")
            install_result = run("apt install -y python3-venv", use_sudo=True, check=False)
            if install_result.returncode == 0:
                print("Retrying virtual environment creation...")
                result = subprocess.run([sys.executable, "-m", "venv", venv_path])
                if result.returncode != 0:
                    print("Failed to create virtual environment after installing python3-venv")
                    if os.path.exists(venv_path):
                        print(f"Cleaning up incomplete virtual environment at {venv_path}")
                        shutil.rmtree(venv_path)
                    sys.exit(1)
            else:
                print("Failed to install python3-venv")
                # Clean up partially created venv directory if it exists
                if os.path.exists(venv_path):
                    print(f"Cleaning up incomplete virtual environment at {venv_path}")
                    shutil.rmtree(venv_path)
                sys.exit(1)

        # Ensure pip is upgraded in the new venv
        venv_python = os.path.join(venv_path, "bin", "python")
        print("Upgrading pip in virtual environment...")
        result = subprocess.run(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip"]
        )
        if result.returncode != 0:
            print("Failed to upgrade pip in virtual environment")
            # Clean up venv directory if pip upgrade fails
            if os.path.exists(venv_path):
                print(
                    f"Cleaning up virtual environment at {venv_path} due to pip failure"
                )
                shutil.rmtree(venv_path)
            sys.exit(1)
    else:
        print(f"Virtual environment already exists at {venv_path}")
    return venv_path


def get_venv_python():
    venv_path = get_venv_path()
    return os.path.join(venv_path, "bin", "python")


def get_venv_pip():
    venv_path = get_venv_path()
    return os.path.join(venv_path, "bin", "pip")


def run_in_venv(command):
    venv_path = get_venv_path()
    activate_cmd = f"source {venv_path}/bin/activate && {command}"
    return subprocess.run(activate_cmd, shell=True, executable="/bin/bash")


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
        "git",
        "python3-pip",
        "python3-venv",
        "build-essential",
        "clang",
        "clangd",
        "wget",
        "zip",
        "unzip",
        "cmake",
        "xorg-dev",
        "libgl1-mesa-dev",
        "libglu1-mesa-dev",
        "libosmesa6-dev",
        "libc++-dev",
        "libeigen3-dev",
        "ffmpeg",
    ]
    return packages


def python_packages():
    return [
        "numpy",
        "pandas",
        "libigl",
        "plyfile",
        "requests",
        "gdown",
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


def dump_python_requirements(path):
    python_reqs = python_packages()
    with open(path, "w") as f:
        f.write("\n".join(python_reqs) + "\n")


def install_lazygit():
    home_bin = os.path.expanduser("~/.local/bin")
    os.makedirs(home_bin, exist_ok=True)
    lazygit_path = os.path.join(home_bin, "lazygit")

    if not os.path.exists(lazygit_path):
        print("installing lazygit")
        cmd = 'curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po \'"tag_name": "v\\K[^"]*\''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        latest_version = result.stdout.strip().replace("v", "")
        print(f"Latest version of lazygit: {latest_version}")
        url = f"https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_{latest_version}_Linux_x86_64.tar.gz"
        subprocess.run(["curl", "-Lo", "lazygit.tar.gz", url], cwd="/tmp")
        subprocess.run(["tar", "xf", "lazygit.tar.gz"], cwd="/tmp")
        shutil.copy("/tmp/lazygit", lazygit_path)
        os.chmod(lazygit_path, 0o755)


def install_nvim():
    local_opt = os.path.expanduser("~/.local/opt")
    local_bin = os.path.expanduser("~/.local/bin")
    nvim_link = os.path.join(local_bin, "nvim")

    # Check if nvim is already installed
    if os.path.exists(nvim_link) or shutil.which("nvim"):
        print("nvim is already installed, skipping...")
    else:
        print("installing nvim")
        os.makedirs(local_opt, exist_ok=True)
        os.makedirs(local_bin, exist_ok=True)

        run(
            "curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz"
        )
        run(f"tar -C {local_opt} -xzf nvim-linux-x86_64.tar.gz")
        if not os.path.exists(nvim_link):
            os.symlink(f"{local_opt}/nvim-linux-x86_64/bin/nvim", nvim_link)

    # Install neovim dependencies
    print("Installing neovim dependencies...")
    nvim_deps = ["fzf", "fd-find", "bat", "ripgrep"]
    run(f"apt install -y {' '.join(nvim_deps)}", use_sudo=True)
    run("~/.cargo/bin/rustup component add rust-analyzer")

    # Create user-local symlinks if commands exist
    if shutil.which("fdfind"):
        fd_link = os.path.join(local_bin, "fd")
        if not os.path.exists(fd_link):
            fdfind = shutil.which("fdfind")
            if fdfind:
                os.symlink(fdfind, fd_link)
    if shutil.which("batcat"):
        bat_link = os.path.join(local_bin, "bat")
        if not os.path.exists(bat_link):
            batcat = shutil.which("batcat")
            if batcat:
                os.symlink(batcat, bat_link)


def install_lazyvim():
    nvim_config_dir = os.path.expanduser("~/.config/nvim")

    # Check if nvim config already exists
    if os.path.exists(nvim_config_dir):
        print(
            f"nvim config already exists at {nvim_config_dir}, skipping LazyVim installation..."
        )
        return

    print("installing lazyvim")
    run("git clone https://github.com/LazyVim/starter ~/.config/nvim")
    run("rm -rf ~/.config/nvim/.git")


def install_fish():
    # Install fish if not present
    if not shutil.which("fish"):
        print("Installing Fish shell...")
        run("apt install -y fish", use_sudo=True)

    print("After installation, run: chsh -s $(which fish)")

    config_dir = os.path.expanduser("~/.config/fish")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "config.fish")

    # Check if fish is installed
    if shutil.which("fish"):
        run("fish -c exit")

        # Check if config.fish is a symlink
        if os.path.islink(config_file):
            print(f"Warning: {config_file} is a symlink. Skipping fish configuration.")
        else:
            # Create config file if it doesn't exist
            if not os.path.exists(config_file):
                with open(config_file, "w") as f:
                    f.write("# Fish configuration\n")

            # Add paths to fish config
            with open(config_file, "a") as f:
                f.write("\n# Added by warmup.py\n")
                f.write("fish_add_path $HOME/.local/bin\n")
                f.write("fish_add_path $HOME/.cargo/bin\n")
                f.write("fish_add_path /usr/local/cuda/bin\n")


def install_oh_my_zsh():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Install zsh if not present
    if not shutil.which("zsh"):
        print("Installing Zsh...")
        run("apt install -y zsh", use_sudo=True)

    if shutil.which("zsh"):
        print("installing oh-my-zsh")
        run(
            'sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended',
            cwd=script_dir,
        )
        run("zsh -c exit")

        zshrc = os.path.expanduser("~/.zshrc")
        venv_path = get_venv_path()
        with open(zshrc, "a") as f:
            f.write("\n# Added by warmup.py\n")
            f.write("export PATH=$HOME/.local/bin:$PATH\n")
            f.write("export PATH=$HOME/.cargo/bin:$PATH\n")
            f.write("export PATH=/usr/local/cuda/bin:$PATH\n")
            f.write(f"export PYTHONPATH={script_dir}:$PYTHONPATH\n")
            f.write("# Activate virtual environment\n")
            f.write(f"source {venv_path}/bin/activate\n")


def install_sdf():
    import time

    pip_path = get_venv_pip()
    max_retries = 3
    for attempt in range(max_retries):
        result = run(
            f"{pip_path} install git+https://github.com/fogleman/sdf.git", check=False
        )
        if result.returncode == 0:
            return
        if attempt < max_retries - 1:
            wait_time = 10 * (attempt + 1)
            print(f"Install sdf failed, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    raise RuntimeError("Failed to install sdf package after multiple retries")


def reinstall_pyopengl():
    pip_path = get_venv_pip()
    run(f"{pip_path} uninstall -y pyopengl")
    run(f"{pip_path} install --force-reinstall --no-deps pyopengl==3.1.5")


def setup():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Install system packages automatically
    print("Installing system packages...")
    packages = list_packages()
    if shutil.which("apt"):
        # Update package list
        print("Running: apt update")
        run("apt update", use_sudo=True)

        # Install packages
        print(f"Installing packages: {' '.join(packages)}")
        run(f"apt install -y {' '.join(packages)}", use_sudo=True)
        print("System packages installed successfully")
    else:
        print("Note: apt not found. Please install these packages manually:")
        sys.exit(1)

    print("")

    # Create virtual environment first
    venv_path = create_venv()

    # Verify pip exists after venv creation
    pip_path = get_venv_pip()
    if not os.path.exists(pip_path):
        print(f"Error: pip not found at {pip_path}")
        print("Trying to bootstrap pip...")
        venv_python = get_venv_python()
        result = subprocess.run([venv_python, "-m", "ensurepip", "--upgrade"])
        if result.returncode != 0 or not os.path.exists(pip_path):
            print("Failed to install pip in virtual environment")
            # Clean up venv directory if pip bootstrap fails
            if os.path.exists(venv_path):
                print(
                    f"Cleaning up virtual environment at {venv_path} due to pip bootstrap failure"
                )
                shutil.rmtree(venv_path)
            sys.exit(1)

    # Check if CUDA is installed
    if not os.path.exists("/usr/local/cuda/bin/nvcc"):
        print("CUDA toolkit not found at /usr/local/cuda")
        print("Installing CUDA toolkit...")
        if shutil.which("apt"):
            # Install CUDA toolkit
            cuda_packages = ["nvidia-cuda-toolkit", "nvidia-cuda-dev"]
            print(f"Installing CUDA packages: {' '.join(cuda_packages)}")
            run(f"apt install -y {' '.join(cuda_packages)}", use_sudo=True)
            print("CUDA packages installed successfully")
    else:
        print("CUDA toolkit found at /usr/local/cuda")

    # Install Python packages in virtual environment
    print("Installing Python packages in virtual environment...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)

    packages = python_packages()
    print(f"Installing {len(packages)} Python packages...")
    result = subprocess.run([pip_path, "install"] + packages, check=True)
    if result.returncode == 0:
        print(f"Successfully installed {len(packages)} packages")

    install_sdf()

    # Node.js installation (user-level)
    print("Installing Node.js via nvm (Node Version Manager)...")
    run(
        "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    )

    # Rust installation (user-level)
    print("Installing Rust...")
    run("curl https://sh.rustup.rs -sSf | sh -s -- -y")
    # Note: rustup installer automatically adds . "$HOME/.cargo/env" to .bashrc

    # Update .bashrc with necessary paths (but NOT .venv)
    bashrc = os.path.expanduser("~/.bashrc")

    # Read existing bashrc content
    try:
        with open(bashrc) as f:
            bashrc_content = f.read()
    except FileNotFoundError:
        bashrc_content = ""

    paths_to_add = []

    # Check and add paths if not already present
    # Check for both variations of the local bin path
    if (
        "$HOME/.local/bin" not in bashrc_content
        and "~/.local/bin" not in bashrc_content
        and "${HOME}/.local/bin" not in bashrc_content
    ):
        paths_to_add.append("export PATH=$HOME/.local/bin:$PATH")

    # Check for PYTHONPATH
    if (
        f"PYTHONPATH={script_dir}" not in bashrc_content
        and f"PYTHONPATH={script_dir}:" not in bashrc_content
    ):
        paths_to_add.append(f"export PYTHONPATH={script_dir}:$PYTHONPATH")

    # Check for CUDA path
    if "/usr/local/cuda/bin" not in bashrc_content:
        paths_to_add.append("export PATH=/usr/local/cuda/bin:$PATH")

    if paths_to_add:
        with open(bashrc, "a") as f:
            f.write("\n# Added by warmup.py\n")
            for path in paths_to_add:
                f.write(f"{path}\n")
        print(f"Added {len(paths_to_add)} path(s) to .bashrc")
    else:
        print("All paths already present in .bashrc")

    # Make paths active in current session
    os.environ["PATH"] = (
        os.path.expanduser("~/.cargo/bin")
        + ":"
        + os.path.expanduser("~/.local/bin")
        + ":"
        + os.environ.get("PATH", "")
    )
    os.environ["PYTHONPATH"] = script_dir + ":" + os.environ.get("PYTHONPATH", "")
    print("Paths activated for current session")

    # Print instructions for .venv usage
    print("\n" + "=" * 60)
    print("PYTHON VIRTUAL ENVIRONMENT:")
    print("=" * 60)
    print(f"Virtual environment created at: {venv_path}")
    print("To use Python packages, always use explicit paths:")
    print(f"  Python: {venv_path}/bin/python")
    print(f"  Pip: {venv_path}/bin/pip")
    print("Or activate manually when needed:")
    print(f"  source {venv_path}/bin/activate")
    print("=" * 60 + "\n")

    reinstall_pyopengl()


def set_tmux():
    # Install tmux if not present
    if not shutil.which("tmux"):
        print("Installing tmux...")
        run("apt install -y tmux", use_sudo=True)

    if not shutil.which("tmux"):
        print("tmux installation failed.")
        return

    tmux_config_file = os.path.expanduser("~/.tmux.conf")

    # Check if it's a symlink
    if os.path.islink(tmux_config_file):
        print(f"Warning: {tmux_config_file} is a symlink. Skipping tmux configuration.")
        return

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
    with open(tmux_config_file, "w") as f:
        for command in tmux_config_commands:
            f.write(command + "\n")


def set_time():
    # Install NTP if not present
    print("Installing NTP...")
    run("apt install -y ntp", use_sudo=True)
    print("NTP service installed and configured")


def start_jupyter():
    import signal
    import time

    run("pkill jupyter-lab", check=False)
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

    # Use user-local Jupyter configuration directory instead of system-wide
    override_file = os.path.expanduser(
        "~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings"
    )
    if not os.path.exists(override_file):
        os.makedirs(os.path.dirname(override_file), exist_ok=True)
        with open(override_file, "w") as f:
            lines = """{
    "theme": "JupyterLab Dark"
}"""
            f.write(lines)

    # Get port from environment or default to 8080
    web_port = os.environ.get("WEB_PORT", "8080")

    # Setup log file
    log_file = "/tmp/jupyter.log"

    # Start JupyterLab in background
    venv_python = get_venv_python()
    command = f"{venv_python} -m jupyterlab -y --allow-root --no-browser --port={web_port} --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir
    env["PYTHONUNBUFFERED"] = "1"

    # Open log file (keep it open for the subprocess, unbuffered)
    with open(log_file, "w", buffering=1) as log_f:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=examples_dir,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        # Signal handler for Ctrl-C
        def signal_handler(sig, frame):
            print("\n\nShutting down JupyterLab...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("JupyterLab shutdown complete")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Monitor log and check for readiness
        jupyter_ready = False
        last_log_size = 0

        print("Starting JupyterLab...")
        sys.stdout.flush()

        try:
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    print("\nJupyterLab process exited unexpectedly")
                    # Show the log file contents for debugging
                    if os.path.exists(log_file):
                        print("\nLog file contents:")
                        print("=" * 60)
                        with open(log_file) as f:
                            print(f.read())
                        print("=" * 60)
                    break

                # Check if jupyter is ready (only if not already confirmed)
                if not jupyter_ready:
                    # Read and display new log lines
                    if os.path.exists(log_file):
                        with open(log_file) as f:
                            f.seek(last_log_size)
                            new_lines = f.read()
                            if new_lines:
                                # Print new log output
                                print(new_lines, end="")
                                sys.stdout.flush()
                                last_log_size = f.tell()

                    try:
                        import urllib.request

                        urllib.request.urlopen(
                            f"http://localhost:{web_port}", timeout=1
                        )
                        jupyter_ready = True
                        # Show final message
                        print()
                        url = f"http://localhost:{web_port}"
                        title = "==== JupyterLab Launched! 🚀 ===="
                        shutdown = "Press Ctrl+C to shutdown"
                        separator = "=" * 32

                        # Calculate spacing to align all lines
                        max_len = max(
                            len(title), len(url), len(shutdown), len(separator)
                        )
                        print(" " * ((max_len - len(title)) // 2) + title)
                        print(" " * ((max_len - len(url)) // 2) + url)
                        print(" " * ((max_len - len(shutdown)) // 2) + shutdown)
                        print(separator)
                        sys.stdout.flush()
                    except (OSError, Exception):
                        pass  # Not ready yet, will try again next iteration

                time.sleep(1)

        except Exception as e:
            print(f"\nError: {e}")
            process.terminate()
            process.wait()


def build_docs():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sphinx_build = os.path.join(get_venv_path(), "bin", "sphinx-build")
    run(f"{sphinx_build} -b html ./ ./_build", cwd=os.path.join(script_dir, "docs"))


def make_docs():
    from frontend import App

    # Generate global parameters documentation
    with open(os.path.join("docs", "global_parameters.rst"), "w") as file:
        file.write(
            export_param_sphinx(App.get_default_param(), title="global parameters")
        )

    # Generate object parameters documentation
    with open(os.path.join("docs", "object_parameters.rst"), "w") as file:
        file.write(export_object_param_sphinx())

    with open(os.path.join("docs", "logs.rst"), "w") as file:
        file.write(export_log_sphinx())
    print("Sphinx .rst files has been exported.")


def export_param_sphinx(param, title="parameters"):
    rst_content = []

    rst_content.append(f"{title}\n")
    rst_content.append("=" * len(title) + "\n\n")

    params_dict = param._param._params

    for name, (value, short_desc, long_desc) in params_dict.items():
        rst_content.append(f"{name}\n")
        rst_content.append("-" * len(name) + "\n\n")

        rst_content.append(".. list-table::\n\n")
        rst_content.append("   * - Parameter\n")
        rst_content.append(f"     - {name}\n")
        rst_content.append("   * - Default Value\n")
        rst_content.append(f"     - {value}\n")
        rst_content.append("   * - Description\n")
        rst_content.append(f"     - {short_desc}\n")
        if long_desc:
            rst_content.append("   * - Details\n")
            rst_content.append(f"     - {long_desc}\n")

        rst_content.append("\n")

    return "".join(rst_content)


def export_object_param_sphinx():
    from frontend._param_ import object_param

    rst_content = []

    title = "object parameters"
    rst_content.append(f"{title}\n")
    rst_content.append("=" * len(title) + "\n\n")

    # Collect parameters from all object types
    obj_types = ["tri", "tet", "rod"]
    all_params = {obj_type: object_param(obj_type) for obj_type in obj_types}

    # Get unique parameter names (they should be the same across types)
    param_names = list(all_params["tri"].keys())

    for name in param_names:
        rst_content.append(f"{name}\n")
        rst_content.append("-" * len(name) + "\n\n")

        rst_content.append(".. list-table::\n\n")
        rst_content.append("   * - Parameter\n")
        rst_content.append(f"     - {name}\n")

        # Show default values for each type
        rst_content.append("   * - Default Value (tri)\n")
        rst_content.append(f"     - {all_params['tri'][name][0]}\n")
        rst_content.append("   * - Default Value (tet)\n")
        rst_content.append(f"     - {all_params['tet'][name][0]}\n")
        rst_content.append("   * - Default Value (rod)\n")
        rst_content.append(f"     - {all_params['rod'][name][0]}\n")

        # Use description from tri (they should be the same)
        short_desc = all_params["tri"][name][1]
        long_desc = all_params["tri"][name][2]

        rst_content.append("   * - Description\n")
        rst_content.append(f"     - {short_desc}\n")
        if long_desc:
            rst_content.append("   * - Details\n")
            rst_content.append(f"     - {long_desc}\n")

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
    # Check architecture - only x86_64 is supported
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        print(f"Error: Architecture '{machine}' is not supported.")
        print("This script only supports x86_64 architecture.")
        sys.exit(1)

    if not os.path.exists(os.path.expanduser("~/.config")):
        os.makedirs(os.path.expanduser("~/.config"))

    args = sys.argv[1:]
    skip_confirmation = False
    while "--skip-confirmation" in args:
        args.remove("--skip-confirmation")
        skip_confirmation = True

    if args:
        mode = args[0]
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
            pip_path = get_venv_pip()
            run(f"{pip_path} install sphinx sphinxawesome-theme sphinx_autobuild")
        elif mode == "docs-build":
            make_docs()
            build_docs()
        elif mode == "requirements":
            dump_python_requirements("requirements.txt")
        elif mode == "all":
            create_clang_config()
            install_nvim()
            install_fish()
            set_tmux()
            install_lazygit()
            install_lazyvim()
    else:
        if not skip_confirmation:
            # Require confirmation before running the full environment setup.
            confirmation = (
                input(
                    "Are you running this in a disposable Docker container or a virtual server? [y/N] "
                )
                .strip()
                .lower()
            )
            if confirmation not in ("y", "yes"):
                print("Exiting without making changes.")
                sys.exit(1)

        create_venv()
        setup()
