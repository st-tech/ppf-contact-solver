# File: _extra_.py
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np


class Extra:
    def load_CIPC_stitch_mesh(
        self, path: str
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
        vertices = []
        faces = []
        stitch_ind = []
        stitch_w = []
        with open(path, "r") as f:
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
