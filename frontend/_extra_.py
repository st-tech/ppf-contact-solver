# File: _extra_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import shutil
import subprocess

import numpy as np


class Extra:
    """Collection of auxiliary helpers that do not belong to any manager.

    Example:
        Access the helpers via :attr:`App.extra` to sparse-clone an
        external dataset and load one of its stitch meshes::

            import os
            from frontend import App, get_cache_dir

            app = App.create("fitting")
            dest = os.path.join(get_cache_dir(), "Codim-IPC")
            app.extra.sparse_clone(
                "https://github.com/ipc-sim/Codim-IPC",
                dest,
                ["Projects/FEMShell/input/dress_knife"],
            )
            stage = os.path.join(dest, "Projects/FEMShell/input/dress_knife/stage.obj")
            V, F, S = app.extra.load_CIPC_stitch_mesh(stage)
    """

    def load_CIPC_stitch_mesh(
        self, path: str
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Load a stitch mesh in the format used by the CIPC paper repository.

        Args:
            path (str): Path to the stitch mesh file.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]: A
            tuple ``(vertices, faces, (stitch_index, stitch_weight))`` where
            vertices have shape ``(#, 3)``, faces have shape ``(#, 3)`` with
            zero-based indices, and each stitch entry has shape ``(#, 4)``.
            The weight row ``[1.0, 1 - w, w, 0.0]`` encodes linear
            interpolation between the second and third stitch vertices.

        Example:
            Load a CIPC-format stitch mesh and register the pieces as
            triangle and stitch assets::

                from frontend import App

                app = App.create("fitting")
                V, F, S = app.extra.load_CIPC_stitch_mesh("dress_stage.obj")
                app.asset.add.tri("dress", V, F)
                app.asset.add.stitch("glue", S)
        """
        vertices = []
        faces = []
        stitch_ind = []
        stitch_w = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v" and len(parts) == 4:
                    x, y, z = map(float, parts[1:])
                    vertices.append([x, y, z])
                elif parts[0] == "f" and len(parts) == 4:
                    face = [int(part.split("/")[0]) for part in parts[1:]]
                    faces.append(face)
                elif parts[0] == "stitch" and len(parts) == 5:
                    idx0, idx1, idx2 = int(parts[1]), int(parts[2]), int(parts[3])
                    w = float(parts[4])
                    stitch_ind.append([idx0, idx1, idx2, idx2])
                    stitch_w.append([1.0, 1 - w, w, 0.0])
        return (
            np.array(vertices),
            np.array(faces) - 1,
            (np.array(stitch_ind), np.array(stitch_w)),
        )

    def sparse_clone(
        self, url: str, dest: str, paths: list[str], delete_exist: bool = False
    ):
        """Fetch a git repository using sparse-checkout.

        Clones ``url`` into ``dest`` if needed, then adds each entry in
        ``paths`` to the sparse-checkout set and checks it out. Already
        present paths are left untouched.

        Args:
            url (str): URL of the git repository.
            dest (str): Destination directory for the clone.
            paths (list[str]): Repository-relative paths to fetch.
            delete_exist (bool): If True, delete ``dest`` before cloning.

        Raises:
            FileNotFoundError: If ``git`` cannot be found on ``PATH``.

        Example:
            Fetch only the FEMShell input subdirectories from the
            upstream CIPC repository into the ppf-cts cache::

                import os
                from frontend import App, get_cache_dir

                app = App.create("fitting")
                dest = os.path.join(get_cache_dir(), "Codim-IPC")
                app.extra.sparse_clone(
                    "https://github.com/ipc-sim/Codim-IPC",
                    dest,
                    ["Projects/FEMShell/input/dress_knife"],
                )
        """
        # Check if git is available
        if shutil.which("git") is None:
            raise FileNotFoundError(
                "git is not found in PATH. The bundled distribution includes MinGit, "
                + "but it may not be in PATH. Please ensure you're running from start.bat "
                + "or add mingit/cmd to your PATH. For manual installation: choco install git"
            )
        if delete_exist and os.path.exists(dest):
            shutil.rmtree(dest)
        if not os.path.exists(dest):
            clone_cmd = [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                dest,
            ]
            print(" ".join(clone_cmd))
            subprocess.run(clone_cmd, check=True)
            set_cmd = ["git", "sparse-checkout", "set"]
            print(" ".join(set_cmd))
            subprocess.run(set_cmd, cwd=dest, check=True)
        for path in paths:
            if not os.path.exists(os.path.join(dest, path)):
                set_cmd = ["git", "sparse-checkout", "add"] + [path]
                print(" ".join(set_cmd))
                subprocess.run(set_cmd, cwd=dest, check=True)
                checkout_cmd = ["git", "checkout"]
                print(" ".join(checkout_cmd))
                subprocess.run(checkout_cmd, cwd=dest, check=True)
            assert os.path.exists(os.path.join(dest, path))
