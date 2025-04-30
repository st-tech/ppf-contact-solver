import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import shutil
import trimesh
from typing import Optional
from PIL import Image
from pyrender import (
    OrthographicCamera,
    SpotLight,
    RenderFlags,
    Mesh,
    Primitive,
    Node,
    Scene,
    OffscreenRenderer,
)

default_args = {
    "variant": "cuda_ad_rgb",
    "max_depth": 12,
    "width": 640,
    "height": 480,
    "fov": 20,
    "camera": None,
    "up": [0, 1, 0],
    "sample_count": 64,
    "tmp_path": os.path.join("/tmp", "tmp_mesh.ply"),
}


def update_default_args(args: dict):
    for key, value in default_args.items():
        if key not in args:
            args[key] = value


class OpenGLRenderer:
    def __init__(self, args: dict = {}):
        update_default_args(args)
        self._r = OffscreenRenderer(
            viewport_width=args["width"], viewport_height=args["height"]
        )

    def __del__(self):
        self._r.delete()

    def render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        seg: np.ndarray,
        face: np.ndarray,
        output: Optional[str],
    ):
        cam = OrthographicCamera(xmag=1.0, ymag=1.0)
        cam_pose = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )
        scene = Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
        rad = np.radians(10)
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rad), -np.sin(rad)],
                [0, np.sin(rad), np.cos(rad)],
            ]
        )
        bounds = np.max(vert, axis=0) - np.min(vert, axis=0)
        center = (bounds / 2) + np.min(vert, axis=0)
        vert = vert - center
        vert = np.dot(vert, rotation_matrix.T)
        vert /= np.max(np.abs(vert))
        vert *= 0.9

        mesh = trimesh.Trimesh(
            vertices=vert, faces=face, vertex_colors=color, process=False
        )
        scene.add_node(
            Node(
                mesh=Mesh.from_trimesh(mesh, smooth=True),
                translation=np.zeros(3),
            ),
        )
        if len(seg):
            positions = np.array([vert[i] for i in seg.ravel()])
            colors = np.array([color[i] for i in seg.ravel()])
            primitive = Primitive(positions=positions, color_0=colors, mode=1)
            scene.add(Mesh(primitives=[primitive]))
        scene.add(cam, pose=cam_pose)
        scene.add(
            SpotLight(
                color=np.ones(3),
                intensity=3.0,
                innerConeAngle=np.pi / 16,
                outerConeAngle=np.pi / 2,
            ),
            pose=cam_pose,
        )
        color, _ = self._r.render(scene, flags=RenderFlags.SKIP_CULL_FACES)  # type: ignore
        image = Image.fromarray(color)
        if output is not None:
            image.save(output)
        return image
