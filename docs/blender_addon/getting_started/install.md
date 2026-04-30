# 🛠️ Install

## Prerequisites

- **Blender 5.0 or newer.** The extension manifest pins `blender_version_min =
  "5.0.0"`; older builds will refuse to enable it.
- **A solver backend.** Build or deploy the engine from
  [st-tech/ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver)
  on any one of: the same machine (simplest), an SSH-reachable Linux host, a
  Docker container, or a Windows workstation. The solver itself requires an
  NVIDIA GPU with CUDA 12.x. See [Connections](../connections/index.md) for
  the full matrix and GPU requirements. The add-on is just a client and runs
  fine on any machine Blender runs on (including macOS).
- **(Optional) paramiko / docker-py.** Needed only for SSH and Docker
  connections. You do not need to install them yourself. When you pick an SSH
  or Docker server type without the module present, the main panel surfaces an
  **Install Paramiko** / **Install Docker-Py** button that installs into the
  add-on's vendored `lib/` directory.

:::{note}
The solver binary itself is not shipped with the add-on. You build or deploy
it separately at the path you point the connection at. The add-on only looks
for a `server.py` entry point there.
:::

## Install the Add-on

The add-on is packaged as a Blender 5.x extension (`blender_manifest.toml`,
schema `1.0.0`, id `ppf_contact_solver`). The recommended path is to register
our static extensions repository so Blender can install and auto-update the
add-on for you. A manual zip install is also documented below for offline or
development use.

### Option A: Add as a Remote Repository (recommended)

This route gets you auto-updates: when a new add-on release is published,
Blender's Get Extensions panel shows it, and one click upgrades.

1. In Blender, open **Edit → Preferences → Get Extensions**.
2. Click the dropdown menu (top-right of the panel) and pick
   **Repositories → + → Add Remote Repository**.
3. Paste this URL into the **URL** field and confirm:

   ```
   https://github.com/st-tech/ppf-contact-solver/releases/download/addon-latest/index.json
   ```

4. Make sure the new repository is enabled, then back in **Get Extensions**
   search for `ZOZO`. The **ZOZO's Contact Solver** add-on should appear;
   click **Install**.
5. Verify by opening the 3D viewport sidebar (`N`). A new tab labeled
   **ZOZO's Contact Solver** should appear.

The URL is permanent. Each new release replaces the `index.json` asset on
the `addon-latest` GitHub Release in place, and the per-version zip lives
on an immutable `addon-YYYY-MM-DD-HHMM` release that the index references
by absolute URL.

### Option B: Install from a Local Checkout (script)

Useful when you are working from a checkout, are offline, or want to install
a build that has not been published as a release yet. The scripts link the
repository's `blender_addon/` directory into Blender's `extensions/user_default`
so edits in the checkout are picked up after a Blender restart, with no
re-zipping.

1. Clone the repository and `cd` into it.
2. Run the script for your platform from the repository root:

   On macOS or Linux:

   ```bash
   ./install-blender-addon.sh
   ```

   On Windows PowerShell:

   ```powershell
   .\install-blender-addon.ps1
   ```

   The script auto-detects your installed Blender version. On a fresh machine
   where Blender has not been launched yet, point it at the binary so it can
   create the prefs directory itself:

   ```bash
   PPF_BLENDER_BIN=/path/to/blender ./install-blender-addon.sh
   ```

   ```powershell
   $env:PPF_BLENDER_BIN = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
   .\install-blender-addon.ps1
   ```

3. Start Blender and enable **ZOZO's Contact Solver** under
   **Edit → Preferences → Add-ons** (search for `ZOZO`), or pass
   `--addons bl_ext.user_default.ppf_contact_solver` on the command line.
4. Verify the sidebar tab as in Option A.

To remove the link later, re-run the script with the uninstall flag:

```bash
./install-blender-addon.sh --uninstall
```

```powershell
.\install-blender-addon.ps1 -Uninstall
```

:::{tip}
If the sidebar tab is missing after install, the add-on probably crashed
while enabling. Open **Window → Toggle System Console** and re-enable the
extension from Preferences to see the traceback.
:::
