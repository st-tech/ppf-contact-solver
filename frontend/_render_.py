import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import os
import numpy as np
import trimesh
from PIL import Image
from pyrender import (
    OrthographicCamera,
    SpotLight,
    RenderFlags,
    Mesh,
    Node,
    Scene,
    OffscreenRenderer,
)


def create_offscreen_renderer(width: int, height: int):
    return OffscreenRenderer(viewport_width=width, viewport_height=height)


def render(
    r: OffscreenRenderer,
    vert: np.ndarray,
    face: np.ndarray,
    color: np.ndarray,
    output: str,
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
    color, _ = r.render(scene, flags=RenderFlags.SKIP_CULL_FACES)  # type: ignore
    image = Image.fromarray(color)
    image.save(output)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", type=str, required=True)
    arg_parser.add_argument("-o", "--output", type=str, required=True)
    args = arg_parser.parse_args()
    r = create_offscreen_renderer(256, 256)

    if os.path.isdir(args.input):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        entries = []
        for file_name in os.listdir(args.input):
            if file_name.endswith(".ply"):
                file_path = os.path.join(args.input, file_name)
                print(f"loading mesh {file_path}")
                mesh = trimesh.load_mesh(file_path)
                vert = mesh.vertices
                face = mesh.faces
                color = mesh.visual.vertex_colors  # type: ignore
                output_file = os.path.join(
                    args.output, f"{os.path.splitext(file_name)[0]}.png"
                )
                entries.append((vert, face, color, output_file))
        for vert, face, color, output_file in entries:
            print(f"rendering mesh to {output_file}")
            render(r, vert, face, color, output_file)
    else:
        print(f"loading mesh {args.input}")
        mesh = trimesh.load_mesh(args.input)
        vert = mesh.vertices
        face = mesh.faces
        color = mesh.visual.vertex_colors  # type: ignore
        print(f"rendering mesh to {args.output}")
        render(r, vert, face, color, args.output)

    r.delete()
