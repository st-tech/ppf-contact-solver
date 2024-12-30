## 🧑‍💻 Setting Up Your Development Environment

Advanced users may be interested in inspecting our 📜 core code to examine how each component ⚙️ contributes to our solver pipeline 🔄. To facilitate this task, we provide a guide below for setting up a comfortable development environment using either 🖥️ [VSCode](https://azure.microsoft.com/en-us/products/visual-studio-code) or ⌨️ [NeoVim](https://neovim.io/).
In fact, this is how we 🚀 develop.

### 🖥️ Complete Installation

First, complete the entire [installation process](../README.md#-getting-started) and keep the Docker container 🚢 running.
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

Nevertheless, for security reasons, we strongly encourage you 👀 review `warmup.py` before running these commands.
The code is not lengthy.
If possible, we also strongly suggest following the official instructions to install them.
These commands exist because this is exactly how we initiate our development environment for all new containers.

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