# 📇 Connection Profiles

A connection profile stores the connection fields of the **Backend
Communicator** panel under a name. Pick a profile to fill in those fields
in one step, and share the `.toml` file so a team connects the same way
(check the file for personal fields first, see [File Format](#file-format)).
One file holds any number of profiles.

## Saving a Profile

Create profiles with the **Save Profile** icon (floppy disk) at the right
end of the profile row. You do not need to write the `.toml` file yourself:
the add-on creates it on the first save.

1. Click **Disconnect** if a connection is live. The whole profile row is
   disabled while connected.
2. Set **Type** and fill in the other fields in the **Connection** box.
3. If the row shows a **Profile** dropdown, click the X icon (**Clear
   Profile**) so that the row shows **Open Profile** and the next save asks
   for a file.
4. Click the floppy-disk icon (**Save Profile**).
5. In the file browser, choose or type the `.toml` file name (the
   suggested name is `connection_profile.toml`). To add an entry to an
   existing file, choose that file. Set **Entry Name** in the file browser's
   options (the default is `Default`), then confirm.
6. The status bar reports `Saved to '<Entry Name>'`. The file is loaded,
   and the **Profile** dropdown shows the new entry.

```{figure} ../images/connections/save_icon.png
:alt: Backend Communicator panel with no profile loaded. The Open Profile button spans the top row, and the floppy-disk Save Profile icon at its right end is highlighted in red. A red label above the icon is cut off at the right edge of the image
:width: 500px

The **Save Profile** icon (floppy disk, highlighted in red) at the right
end of the profile row, shown with no file loaded.
```

With a file loaded and an entry selected, the same icon overwrites that
entry at once, with no file browser, and reports `Saved to '<name>'`. An
**Entry Name** that already exists in the chosen file replaces that entry.

## The Profile Row

The row at the top of the **Backend Communicator** panel has two states.

| State | What the row shows |
| ----- | ------------------ |
| No file loaded | An **Open Profile** button and the **Save Profile** icon. |
| File loaded | The **Profile** dropdown, followed by four icon buttons. |

The add-on remembers the loaded file in the `.blend`. If the file is moved
or deleted, the row goes back to the no-file state.

```{figure} ../images/connections/profile_row.png
:alt: The profile row with no file loaded: an Open Profile button with a folder icon, and the floppy-disk Save Profile icon at the right end of the row
:width: 500px

The profile row with no file loaded. **Open Profile** picks a `.toml`
file. Once a file is loaded, the row shows the **Profile** dropdown and
the four icon buttons listed below.
```

With a file loaded, the four buttons are icons without text:

| Icon | Button | Effect |
| ---- | ------ | ------ |
| Folder | **Open Profile** | Pick a `.toml` file. The **Profile** dropdown lists its entries, and the first entry in the sorted list is selected and applied. |
| X | **Clear Profile** | Forget the loaded file. The connection fields keep their values. |
| Circular arrows | **Reload Profile** | Re-read the file and re-apply the fields the selected entry names, which restores those fields to the file's values. |
| Floppy disk | **Save Profile** | Overwrite the selected entry with the current field values. With no entry selected, ask for a file and an **Entry Name** (see [Saving a Profile](#saving-a-profile)). |

**Save Profile** writes what is in the panel at that moment. The other
entries stay in the file when it holds entry tables and nothing else, each
with plain values (see [File Format](#file-format)).

:::{warning}
**Save Profile** rewrites the whole file from the entries the add-on can
read, and removes comments and formatting. Keep a backup of a file you have
edited by hand.

Two kinds of content do not survive a save, and the **Profile** dropdown
gives no warning for either:

- A top-level key that is not a table, such as `version = 2` above the first
  entry. The save fails with an error and leaves the file cut short, so
  entries are lost. The dropdown also lists such a key as if it were an
  entry.
- A table nested inside an entry, such as `extra = { x = 1 }` or a
  `[GPU01.extra]` table. The save writes it back in a form that is not valid
  TOML, so the add-on cannot read the file.

Remove such keys and tables before you save, or save into a different file.

If the **Profile** dropdown reads **(No Profile)** for a file that should
hold entries, the add-on could not read the file: it is empty or is not
valid TOML, and the add-on does not report parse errors. Do not save into
that file, because every entry the add-on could not read is lost.
:::

## File Format

Open a profile file in an editor to inspect, compare or share entries.
Each profile is a top-level table whose name is the entry name. A name made
of ASCII letters, digits, `_` and `-` needs no quotes. Quote any other name,
as in `["GPU01 via bastion"]`.

The add-on quotes a name that contains a space or punctuation when it saves.
A name made only of letters (non-ASCII ones included), digits, `_` and `-`
is written without quotes, so `Büro` is written as `[Büro]`. TOML allows only
ASCII letters in an unquoted name, so the file does not parse and the add-on
cannot read it (see the warning under
[The Profile Row](#the-profile-row)). Use an ASCII **Entry Name**. Quotes
added by hand to such a name are removed again by the next save, so rename
an entry that already has one.

Inside the table are one required key, `type`, and up to fifteen connection
fields:

| TOML key | Panel field | Notes |
| -------- | ----------- | ----- |
| `type` | **Type** | Required. One of `SSH`, `SSH Command`, `Docker`, `Docker over SSH`, `Docker over SSH Command`, `Windows Native`, `macOS Native`, `Linux Native`. |
| `host` | **Host** | SSH host or alias, for SSH and Docker over SSH. The two Command types take the host from `command`. |
| `port` | **Port** | SSH port, for SSH and Docker over SSH. |
| `username` | **User** | SSH user, for SSH and Docker over SSH. |
| `key_path` | **SSH Key** | Private key path, for SSH and Docker over SSH. The Command types take `-i` from `command`. Write the full path: the panel does not expand a leading `~` in this field, and a `~` path makes **Connect** fail. Empty uses the `IdentityFile` from `~/.ssh/config`. |
| `proxy_jump` | **Proxy Jump** | Jump host to tunnel through, as `ssh -J` takes it, for SSH and Docker over SSH. Empty falls back to `~/.ssh/config`. The Command types ignore this key: put `-J` in `command`. |
| `command` | **SSH Command** | Raw `ssh ...` string for the two Command types. A leading `~` after `-i` is expanded. |
| `container` | **Container** | Docker container name. |
| `remote_path` | **Remote Path** | Solver directory on the solver host, for SSH and SSH Command. |
| `docker_path` | **Container Path** | Solver directory inside the container, for the Docker types. |
| `win_native_path` | **Solver Path** | Windows solver root, for Windows Native. |
| `mac_native_path` | **Solver Path** | macOS solver root, for macOS Native. |
| `linux_native_path` | **Solver Path** | Linux solver root, for Linux Native. |
| `docker_port` | **Docker Port** | Server TCP port (1024-65535). Applies to every type, although the panel draws the field only for the Docker types. |
| `solver_gpu` | **GPU** | CUDA device index of the dropdown entry, or `-1` for Automatic. The launch uses `solver_gpu_uuid`. |
| `solver_gpu_uuid` | **GPU** | Stable UUID of the selected GPU. This is the identity used at **Start Server on Remote**. An entry that has `solver_gpu` but no `solver_gpu_uuid` clears the saved UUID. |

The three solver-path keys (`win_native_path`, `mac_native_path` and
`linux_native_path`) name a folder on the machine Blender runs on. Write an
absolute path, or `//` followed by a path relative to the `.blend`. A
leading `~` is not expanded.

The **GPU** dropdown is drawn only after **Connect**, and not on
**macOS Native** or with **Compute Device** on **CPU** (see
{ref}`Picking a GPU <gpu-picker>`), but the profile row is disabled while
connected. To store a different GPU, pick it after **Connect**, click
**Disconnect**, then click **Save Profile**. The entry stores the GPU that
was picked last.

**Save Profile** writes `type` and all fifteen fields for every entry,
whatever its type. The fields a type does not use hold whatever the panel
holds for them. Only the fields in the table are saved: **Project Name**,
**Compute Device** and **GPU Backend** are not, so loading an entry leaves
them as they are (see
{ref}`Choosing the build <choosing-the-build>`).

Keys the add-on does not recognize are ignored when an entry is loaded.
**Save Profile** drops them from the entry it overwrites and keeps them in
the other entries, as long as each holds a plain value (see the warning
under [The Profile Row](#the-profile-row)).

:::{note}
An entry written by **Save Profile** can hold your user name, the path of
your SSH key, a jump host and an SSH command. Remove those fields before you
share a file. See
[Credential Files on Disk](../security.md#credential-files-on-disk).
:::

## Example

A file written by **Save Profile** lists every key. The examples below show
only the keys that matter for each type. That is enough, because loading an
entry changes only the fields it names (see
[Loading a Profile](#loading-a-profile)).

```toml
# connection_profile.toml: one entry per environment
["Workstation Linux"]
type = "Linux Native"
linux_native_path = "/home/alice/ppf-contact-solver"
docker_port = 9090

["Workstation Windows"]
type = "Windows Native"
win_native_path = "C:\\Users\\alice\\ppf-win"
docker_port = 9090

["Workstation Mac"]
type = "macOS Native"
mac_native_path = "/Users/alice/ppf-contact-solver-macos"
docker_port = 9090

[LocalDocker]
type = "Docker"
container = "ppf-contact-solver"
docker_path = "/root/ppf-contact-solver"
docker_port = 9090

[GPU01]
type = "SSH"
host = "gpu01.example.com"
port = 22
username = "alice"
key_path = "/home/alice/.ssh/id_ed25519"
remote_path = "/home/alice/ppf-contact-solver"
docker_port = 9090

["GPU01 via command"]
type = "SSH Command"
command = "ssh -p 22 -i ~/.ssh/id_ed25519 alice@gpu01.example.com"
remote_path = "/home/alice/ppf-contact-solver"
docker_port = 9090

["GPU01 via bastion"]
type = "SSH"
host = "gpu01.internal"
username = "alice"
key_path = "/home/alice/.ssh/id_ed25519"
proxy_jump = "bastion.example.com"
remote_path = "/home/alice/ppf-contact-solver"
docker_port = 9090

["GPU01 Docker"]
type = "Docker over SSH"
host = "gpu01.example.com"
port = 22
username = "alice"
key_path = "/home/alice/.ssh/id_ed25519"
container = "ppf-contact-solver"
docker_path = "/root/ppf-contact-solver"
docker_port = 9090

["GPU01 Docker via command"]
type = "Docker over SSH Command"
command = "ssh -i ~/.ssh/id_ed25519 alice@gpu01.example.com"
container = "ppf-contact-solver"
docker_path = "/root/ppf-contact-solver"
docker_port = 9090
```

In the `.toml` file, write each backslash of a Windows path twice (`\\`),
as in the Windows path above, or use forward slashes.

## Loading a Profile

1. Click **Disconnect** if a connection is live. The profile row is
   disabled while connected.
2. Click **Open Profile** and pick the `.toml` file. The **Profile**
   dropdown lists the entries of the file, sorted by name, and the first
   entry is selected and applied at once.
3. To use another entry, pick it in the dropdown. Its values are applied as
   soon as you pick it.
4. Check the fields. An entry changes only the fields it names, so
   switching from an SSH entry to a Docker entry leaves **Host** and
   **User** at their old values.
5. Edit any field you like, then **Connect** as usual. Your edits stay in
   the panel. **Reload Profile** re-applies the fields the entry names (every
   field, for an entry written by **Save Profile**), and **Save Profile**
   writes the panel values to the selected entry.

An entry whose `type` is missing or not recognized changes nothing, and
the add-on reports no error.

## Related Profile Types

The same four icons (folder, X, circular arrows and floppy disk) manage
three other kinds of profile, each in its own file:

- [Scene profiles](../workflow/params/scene.md#scene-profiles), in the
  **Scene Configuration** panel.
- [Material profiles](../workflow/params/material.md#material-profiles),
  one per group.
- [Pin operation profiles](../workflow/constraints/pins.md#saving-operations-to-a-profile),
  under the **Operations:** label of each pin.

The four icons behave the same way, apart from the disabled-while-connected
rule, which applies only to connection profiles.

:::{admonition} Under the hood
:class: toggle

**Operators**

| Button | Operator `bl_idname` |
| ------ | -------------------- |
| **Open Profile** | `ssh.open_profile` |
| **Clear Profile** | `ssh.clear_profile` |
| **Reload Profile** | `ssh.reload_profile` |
| **Save Profile** | `ssh.save_profile` |

**Save behavior**

**Save Profile** loads the file on disk, replaces only the selected entry
with the on-screen values, and writes the whole file back. The writer
handles scalars, lists and arrays of tables. It does not handle a table
nested inside an entry, which it writes as a Python dictionary literal that
is not valid TOML, or a top-level key that is not a table, which raises an
error after the file has been opened for writing. A file that does not parse
loads as empty, so saving into it writes a file holding only the new
entry.

**`type` validation**

The `type` value must match one of the eight type strings in the
**File Format** table. The spelling `Local` is also accepted and is read
as the Native type of your platform, and a `local_path` key is read as
the solver-path key of your platform (`win_native_path`, `mac_native_path`
or `linux_native_path`). Any other value, or a missing `type`,
makes the load return before it touches anything. The selected connection
type and every field on screen stay as they were, none of the entry's
values are applied, and no error is reported.
:::
