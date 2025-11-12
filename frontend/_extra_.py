# File: _extra_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import shutil
import subprocess

import numpy as np


class Extra:
    """Extra class. Use this to perform extra operations."""

    def load_CIPC_stitch_mesh(
        self, path: str
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Load a stitch mesh data in the CIPC paper repository.

        Args:
            path (str): The path to the stitch mesh data.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]: A tuple
            containing the vertices (#x3), faces (#x3), and stitch data (index #x3 and weight #x2).
            The weight encodes the liner interpolation between the last two vertices.
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
                    stitch_ind.append([idx0, idx1, idx2])
                    stitch_w.append([1 - w, w])
        return (
            np.array(vertices),
            np.array(faces) - 1,
            (np.array(stitch_ind), np.array(stitch_w)),
        )

    def sparse_clone(
        self, url: str, dest: str, paths: list[str], delete_exist: bool = False
    ):
        """Fetch a git repository with sparse-checkout

        Args:
            url (str): The URL to the git repository.
            dest (str): The destination directory to clone the repository.
            paths (list[str]): The list of paths to fetch.
            delete_exist (bool): If True, delete the existing repository.
        """
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
