# File: _render_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import ctypes
import os
import shutil
from typing import Optional

import numpy as np
from PIL import Image

from ._utils_ import get_cache_dir

# Set up OSMesa for headless software rendering (Linux only)
# Windows does not have OSMesa, so OpenGL rendering is disabled there
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

# Try to import OpenGL with OSMesa for headless rendering
try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_ELEMENT_ARRAY_BUFFER,
        GL_FLOAT,
        GL_FRAGMENT_SHADER,
        GL_LESS,
        GL_LINES,
        GL_RGBA,
        GL_STATIC_DRAW,
        GL_TRIANGLES,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_INT,
        GL_VERTEX_SHADER,
        glAttachShader,
        glBindBuffer,
        glBindVertexArray,
        glBufferData,
        glClear,
        glClearColor,
        glCompileShader,
        glCreateProgram,
        glCreateShader,
        glDeleteBuffers,
        glDeleteProgram,
        glDeleteShader,
        glDeleteVertexArrays,
        glDepthFunc,
        glDrawArrays,
        glDrawElements,
        glEnable,
        glEnableVertexAttribArray,
        glGenBuffers,
        glGenVertexArrays,
        glGetAttribLocation,
        glGetUniformLocation,
        glLineWidth,
        glLinkProgram,
        glReadPixels,
        glShaderSource,
        glUniform3f,
        glUniformMatrix4fv,
        glUseProgram,
        glVertexAttribPointer,
        glViewport,
    )
    from OpenGL.osmesa import (
        OSMESA_CONTEXT_MAJOR_VERSION,
        OSMESA_CONTEXT_MINOR_VERSION,
        OSMESA_CORE_PROFILE,
        OSMESA_DEPTH_BITS,
        OSMESA_FORMAT,
        OSMESA_PROFILE,
        OSMESA_RGBA,
        OSMesaCreateContextAttribs,
        OSMesaDestroyContext,
        OSMesaMakeCurrent,
    )
    from OpenGL import arrays

    OPENGL_READY = True
except (ImportError, AttributeError):
    OPENGL_READY = False

default_args = {
    "variant": "cuda_ad_rgb",
    "max_depth": 12,
    "width": 640,
    "height": 480,
    "fov": 20,
    "camera": None,
    "up": [0, 1, 0],
    "sample_count": 64,
    "tmp_path": os.path.join(get_cache_dir(), "tmp_mesh.ply"),
}


def update_default_args(args: dict):
    for key, value in default_args.items():
        if key not in args:
            args[key] = value


# Vertex shader with simple lighting (for triangles)
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

out vec3 fragColor;
out vec3 fragNormal;
out vec3 fragPos;

uniform mat4 mvp;
uniform mat4 model;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    fragColor = color;
    fragNormal = mat3(model) * normal;
    fragPos = vec3(model * vec4(position, 1.0));
}
"""

# Fragment shader with Phong-like lighting (for triangles)
FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
in vec3 fragNormal;
in vec3 fragPos;

out vec4 outColor;

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;

void main() {
    vec3 norm = normalize(fragNormal);
    // Two-sided lighting
    float diff = abs(dot(norm, lightDir));
    vec3 diffuse = diff * lightColor;
    vec3 result = (ambientColor + diffuse) * fragColor;
    outColor = vec4(result, 1.0);
}
"""

# Simple vertex shader for lines (no lighting)
LINE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 fragColor;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    fragColor = color;
}
"""

# Simple fragment shader for lines
LINE_FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;

out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
"""


def _compile_shader(source: str, shader_type: int) -> int:
    """Compile a shader from source."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader


def _create_program(vertex_src: str, fragment_src: str) -> int:
    """Create a shader program from vertex and fragment shaders."""
    vertex_shader = _compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = _compile_shader(fragment_src, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from faces."""
    normals = np.zeros_like(vertices)

    # Compute face normals and accumulate to vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)

    # Accumulate face normals to vertex normals
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1
    normals = normals / lengths

    return normals


def _ortho_matrix(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    """Create an orthographic projection matrix."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


def _rotation_x(angle: float) -> np.ndarray:
    """Create a rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


class OpenGLRenderer:
    """Custom OpenGL renderer using OSMesa for headless rendering."""

    def __init__(self, args: Optional[dict] = None):
        if not OPENGL_READY:
            raise RuntimeError(
                "OpenGL rendering is not available. "
                "Install OSMesa: apt install libosmesa6-dev\n"
                "Or skip rendering with: session.export.animation(options={'skip_render': True})"
            )
        if args is None:
            args = {}
        update_default_args(args)
        self._width = args["width"]
        self._height = args["height"]
        self._context = None
        self._buffer = None

        # Use OSMesa for headless software rendering
        attrs = arrays.GLintArray.asArray([
            OSMESA_FORMAT, OSMESA_RGBA,
            OSMESA_DEPTH_BITS, 24,
            OSMESA_PROFILE, OSMESA_CORE_PROFILE,
            OSMESA_CONTEXT_MAJOR_VERSION, 3,
            OSMESA_CONTEXT_MINOR_VERSION, 3,
            0
        ])
        self._context = OSMesaCreateContextAttribs(attrs, None)
        if not self._context:
            raise RuntimeError("Failed to create OSMesa context")

        # Create buffer for rendering
        self._buffer = arrays.GLubyteArray.zeros((self._height, self._width, 4))

        # Make context current
        if not OSMesaMakeCurrent(self._context, self._buffer, GL_UNSIGNED_BYTE, self._width, self._height):
            raise RuntimeError("Failed to make OSMesa context current")

        # Setup OpenGL
        glViewport(0, 0, self._width, self._height)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glClearColor(1.0, 1.0, 1.0, 1.0)  # White background

        # Create shader programs
        self._program = _create_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self._line_program = _create_program(LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER)

    def __del__(self):
        if hasattr(self, "_context") and self._context:
            try:
                OSMesaDestroyContext(self._context)
            except (TypeError, AttributeError):
                pass  # OpenGL already cleaned up during interpreter shutdown

    def render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        seg: np.ndarray,
        face: np.ndarray,
        output: str | None,
    ):
        """Render a mesh with vertex colors."""
        # Make context current
        OSMesaMakeCurrent(self._context, self._buffer, GL_UNSIGNED_BYTE, self._width, self._height)

        # Transform vertices: center, rotate, normalize
        rad = np.radians(10)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)],
        ])
        bounds = np.max(vert, axis=0) - np.min(vert, axis=0)
        center = (bounds / 2) + np.min(vert, axis=0)
        vert = vert - center
        vert = np.dot(vert, rotation_matrix.T)
        max_extent = np.max(np.abs(vert))
        if max_extent > 0:
            vert = vert / max_extent * 0.9

        # Prepare vertex data
        vert = vert.astype(np.float32)
        color = color.astype(np.float32)
        if color.shape[1] == 4:
            color = color[:, :3]  # Remove alpha if present

        # Setup matrices (shared between triangle and line rendering)
        aspect = self._width / self._height
        if aspect >= 1:
            proj = _ortho_matrix(-aspect, aspect, -1, 1, -10, 10)
        else:
            proj = _ortho_matrix(-1, 1, -1/aspect, 1/aspect, -10, 10)
        model = np.eye(4, dtype=np.float32)
        mvp = proj @ model

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        has_faces = len(face) > 0
        has_lines = len(seg) > 0

        # Render triangles if present
        if has_faces:
            # Compute normals for triangles
            normals = _compute_normals(vert, face)
            normals = normals.astype(np.float32)

            # Vertex data with normals (position + color + normal)
            vertex_data = np.hstack([vert, color, normals]).astype(np.float32)
            face_arr = face.astype(np.uint32)

            # Create VAO for triangles
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            vbo = glGenBuffers(1)
            ebo = glGenBuffers(1)

            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_arr.nbytes, face_arr, GL_STATIC_DRAW)

            # Setup vertex attributes for triangle shader
            stride = 9 * 4  # 9 floats * 4 bytes
            glUseProgram(self._program)

            pos_loc = glGetAttribLocation(self._program, "position")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))

            color_loc = glGetAttribLocation(self._program, "color")
            glEnableVertexAttribArray(color_loc)
            glVertexAttribPointer(color_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(3 * 4))

            normal_loc = glGetAttribLocation(self._program, "normal")
            glEnableVertexAttribArray(normal_loc)
            glVertexAttribPointer(normal_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(6 * 4))

            # Set uniforms
            mvp_loc = glGetUniformLocation(self._program, "mvp")
            glUniformMatrix4fv(mvp_loc, 1, True, mvp)

            model_loc = glGetUniformLocation(self._program, "model")
            glUniformMatrix4fv(model_loc, 1, True, model)

            light_dir = np.array([0.5, 0.5, 1.0], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)
            light_dir_loc = glGetUniformLocation(self._program, "lightDir")
            glUniform3f(light_dir_loc, *light_dir)

            light_color_loc = glGetUniformLocation(self._program, "lightColor")
            glUniform3f(light_color_loc, 0.7, 0.7, 0.7)

            ambient_loc = glGetUniformLocation(self._program, "ambientColor")
            glUniform3f(ambient_loc, 0.3, 0.3, 0.3)

            # Draw triangles
            glDrawElements(GL_TRIANGLES, len(face_arr) * 3, GL_UNSIGNED_INT, None)

            # Cleanup triangle buffers
            glDeleteBuffers(1, [vbo])
            glDeleteBuffers(1, [ebo])
            glDeleteVertexArrays(1, [vao])

        # Render line segments if present
        if has_lines:
            # Build line vertex data from segments
            line_positions = []
            line_colors = []
            for s in seg:
                line_positions.append(vert[s[0]])
                line_positions.append(vert[s[1]])
                line_colors.append(color[s[0]])
                line_colors.append(color[s[1]])

            line_positions = np.array(line_positions, dtype=np.float32)
            line_colors = np.array(line_colors, dtype=np.float32)
            line_vertex_data = np.hstack([line_positions, line_colors]).astype(np.float32)

            # Create VAO for lines
            line_vao = glGenVertexArrays(1)
            glBindVertexArray(line_vao)

            line_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, line_vbo)
            glBufferData(GL_ARRAY_BUFFER, line_vertex_data.nbytes, line_vertex_data, GL_STATIC_DRAW)

            # Setup vertex attributes for line shader
            stride = 6 * 4  # 6 floats * 4 bytes (position + color)
            glUseProgram(self._line_program)

            pos_loc = glGetAttribLocation(self._line_program, "position")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))

            color_loc = glGetAttribLocation(self._line_program, "color")
            glEnableVertexAttribArray(color_loc)
            glVertexAttribPointer(color_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(3 * 4))

            # Set uniforms
            mvp_loc = glGetUniformLocation(self._line_program, "mvp")
            glUniformMatrix4fv(mvp_loc, 1, True, mvp)

            # Draw lines
            glLineWidth(2.0)
            glDrawArrays(GL_LINES, 0, len(line_positions))

            # Cleanup line buffers
            glDeleteBuffers(1, [line_vbo])
            glDeleteVertexArrays(1, [line_vao])

        # Read pixels
        pixels = glReadPixels(0, 0, self._width, self._height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (self._width, self._height), pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        if output is not None:
            image.save(output)
        return image


class MitsubaRenderer:
    def __init__(self, args: Optional[dict] = None):
        if args is None:
            args = {}
        assert shutil.which("mitsuba") is not None
        update_default_args(args)
        self._args = args

    def __del__(self):
        if os.path.exists(self._args["tmp_path"]):
            os.remove(self._args["tmp_path"])

    def render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        seg: np.ndarray,
        face: np.ndarray,
        path: str,
    ):
        import mitsuba as mi

        mi.set_variant(self._args["variant"])
        self._export_ply(self._args["tmp_path"], vert, color, face)
        mesh = mi.load_dict(
            {
                "type": "ply",
                "filename": self._args["tmp_path"],
                "bsdf": {
                    "type": "twosided",
                    "bsdf": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "mesh_attribute",
                            "name": "vertex_color",
                        },
                    },
                },
            }
        )
        if len(seg):
            print("Mitsuba does not support line primitives with varying colors.")

        bounds = np.max(vert, axis=0) - np.min(vert, axis=0)
        width, height = self._args["width"], self._args["height"]
        if type(self._args["camera"]) is dict:
            origin = self._args["camera"]["origin"]
            target = self._args["camera"]["target"]
        else:
            rat = width / height
            target = (bounds / 2) + np.min(vert, axis=0)
            origin = target + np.max(bounds) * 0.5 * rat * np.array([0, 1, 5])
            target, origin = target.tolist(), origin.tolist()
        scene = mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path", "max_depth": self._args["max_depth"]},
                "sensor": {
                    "type": "perspective",
                    "fov": self._args["fov"],
                    "fov_axis": "x",
                    "to_world": mi.ScalarTransform4f().look_at(
                        origin=mi.ScalarPoint3f(origin),
                        target=mi.ScalarPoint3f(target),
                        up=mi.ScalarPoint3f(self._args["up"]),
                    ),
                    "sampler": {
                        "type": "independent",
                        "sample_count": self._args["sample_count"],
                    },
                    "film": {
                        "type": "hdrfilm",
                        "width": width,
                        "height": height,
                    },
                },
                "emitter-1": {
                    "type": "directional",
                    "direction": [1, -1, -1],
                    "irradiance": {"type": "rgb", "value": 2.0},
                },
                "emitter-2": {
                    "type": "constant",
                    "radiance": {"type": "rgb", "value": 0.25},
                },
                "wall": {
                    "type": "rectangle",
                    "emitter": {
                        "type": "area",
                        "radiance": {"type": "rgb", "value": 0.75},
                    },
                    "to_world": mi.ScalarTransform4f()
                    .scale(mi.ScalarPoint3f([1000, 1000, 1]))
                    .translate(mi.ScalarPoint3f([0, 0, -5 * bounds[2]])),
                },
                "my-mesh": mesh,
            }
        )
        image = mi.render(scene)  # pyright: ignore
        mi.util.write_bitmap(path, image)

    def _export_ply(
        self, path: str, vertex: np.ndarray, color: np.ndarray, face: np.ndarray
    ):
        with open(path, "wb") as ply_file:
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(f"element vertex {vertex.shape[0]}\n".encode())
            ply_file.write(b"property float x\n")
            ply_file.write(b"property float y\n")
            ply_file.write(b"property float z\n")
            ply_file.write(b"property float red\n")
            ply_file.write(b"property float green\n")
            ply_file.write(b"property float blue\n")
            ply_file.write(f"element face {face.shape[0]}\n".encode())
            ply_file.write(b"property list uchar int vertex_indices\n")
            ply_file.write(b"end_header\n")
            for i in range(vertex.shape[0]):
                assert vertex[i].shape == (3,)
                assert color[i].shape == (3,)
                ply_file.write(vertex[i].astype(np.float32).tobytes())
                ply_file.write(color[i].astype(np.float32).tobytes())
            for i in range(face.shape[0]):
                assert face[i].shape == (3,)
                ply_file.write(np.array([3], dtype=np.uint8).tobytes())
                ply_file.write(face[i].astype(np.int32).tobytes())


if __name__ == "__main__":
    import argparse
    import trimesh

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-e", "--engine", type=str, required=True)
    arg_parser.add_argument("-i", "--input", type=str, required=True)
    arg_parser.add_argument("-o", "--output", type=str, required=True)
    arg_parser.add_argument("--origin", type=float, nargs=3)
    arg_parser.add_argument("--target", type=float, nargs=3)
    arg_parser.add_argument("--width", type=int, default=640)
    arg_parser.add_argument("--height", type=int, default=480)
    arg_parser.add_argument("--sample-count", type=int, default=64)
    args = arg_parser.parse_args()
    if args.origin is not None and args.target is not None:
        options = {
            "camera": {
                "origin": args.origin,
                "target": args.target,
            },
        }
    else:
        options = {}
    options["width"] = args.width
    options["height"] = args.height
    options["sample_count"] = args.sample_count

    if args.engine == "opengl":
        r = OpenGLRenderer(options)
    elif args.engine == "mitsuba":
        r = MitsubaRenderer(options)
    else:
        raise ValueError(f"unsupported engine {args.engine}")

    if os.path.isdir(args.input):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        entries = []
        for filename in os.listdir(args.input):
            if filename.endswith(".ply"):
                filepath = os.path.join(args.input, filename)
                segpath = filepath.replace(".ply", ".seg")
                if os.path.exists(segpath):
                    seg = np.loadtxt(segpath, dtype=np.uint32).reshape((-1, 2))
                else:
                    seg = np.zeros((0, 2), dtype=np.uint32)
                print(f"loading mesh {filepath}")
                mesh = trimesh.load_mesh(filepath)
                vert = mesh.vertices
                face = mesh.faces
                color = mesh.visual.vertex_colors  # type: ignore
                color = color[:, :3] / 255.0
                output_file = os.path.join(
                    args.output, f"{os.path.splitext(filename)[0]}.png"
                )
                entries.append((vert, color, seg, face, output_file))
        for vert, color, seg, face, output_file in entries:
            print(f"rendering mesh to {output_file}")
            r.render(vert, color, seg, face, output_file)
    else:
        assert args.input.endswith(".ply")
        print(f"loading mesh {args.input}")
        mesh = trimesh.load_mesh(args.input)
        vert = mesh.vertices
        segpath = args.input.replace(".ply", ".seg")
        if os.path.exists(segpath):
            seg = np.loadtxt(segpath, dtype=np.uint32).reshape((-1, 2))
        else:
            seg = np.zeros((0, 2), dtype=np.uint32)
        face = mesh.faces
        color = mesh.visual.vertex_colors  # type: ignore
        color = color[:, :3] / 255.0
        print(f"rendering mesh to {args.output}")
        r.render(vert, color, seg, face, args.output)
