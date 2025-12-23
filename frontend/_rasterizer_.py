# File: _rasterizer_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""
Pure software rasterizer using NumPy and Numba.
Lightweight headless rendering without heavy dependencies.
"""

import numpy as np
from numba import njit, prange
from PIL import Image
from typing import Optional


@njit(parallel=True, cache=True)
def _rasterize_triangles(
    framebuffer: np.ndarray,
    depth_buffer: np.ndarray,
    screen_verts: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    light_dir: np.ndarray,
    ambient: float,
) -> None:
    """Rasterize triangles with depth testing and simple lighting.

    Args:
        framebuffer: RGBA image buffer (H, W, 4) uint8
        depth_buffer: Depth buffer (H, W) float32
        screen_verts: Screen-space vertices (N, 4) [x, y, z, w]
        colors: Vertex colors (N, 3) float32
        normals: Vertex normals (N, 3) float32
        faces: Triangle indices (F, 3) int32
        light_dir: Normalized light direction (3,) float32
        ambient: Ambient light intensity
    """
    height, width = framebuffer.shape[:2]

    for fi in prange(len(faces)):
        i0, i1, i2 = faces[fi]

        # Get screen coordinates
        x0, y0, z0 = screen_verts[i0, 0], screen_verts[i0, 1], screen_verts[i0, 2]
        x1, y1, z1 = screen_verts[i1, 0], screen_verts[i1, 1], screen_verts[i1, 2]
        x2, y2, z2 = screen_verts[i2, 0], screen_verts[i2, 1], screen_verts[i2, 2]

        # Bounding box
        min_x = max(0, int(min(x0, x1, x2)))
        max_x = min(width - 1, int(max(x0, x1, x2)) + 1)
        min_y = max(0, int(min(y0, y1, y2)))
        max_y = min(height - 1, int(max(y0, y1, y2)) + 1)

        # Precompute edge vectors for barycentric coords
        v0x, v0y = x2 - x0, y2 - y0
        v1x, v1y = x1 - x0, y1 - y0

        denom = v0x * v1y - v1x * v0y
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        # Iterate over bounding box
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                # Compute barycentric coordinates
                v2x = px - x0
                v2y = py - y0

                u = (v2x * v1y - v1x * v2y) * inv_denom
                v = (v0x * v2y - v2x * v0y) * inv_denom
                w = 1.0 - u - v

                # Check if inside triangle
                if u >= 0 and v >= 0 and w >= 0:
                    # Interpolate depth
                    z = w * z0 + v * z1 + u * z2

                    # Depth test
                    if z < depth_buffer[py, px]:
                        depth_buffer[py, px] = z

                        # Interpolate color
                        r = w * colors[i0, 0] + v * colors[i1, 0] + u * colors[i2, 0]
                        g = w * colors[i0, 1] + v * colors[i1, 1] + u * colors[i2, 1]
                        b = w * colors[i0, 2] + v * colors[i1, 2] + u * colors[i2, 2]

                        # Interpolate normal
                        nx = w * normals[i0, 0] + v * normals[i1, 0] + u * normals[i2, 0]
                        ny = w * normals[i0, 1] + v * normals[i1, 1] + u * normals[i2, 1]
                        nz = w * normals[i0, 2] + v * normals[i1, 2] + u * normals[i2, 2]

                        # Normalize
                        n_len = np.sqrt(nx * nx + ny * ny + nz * nz)
                        if n_len > 1e-10:
                            nx /= n_len
                            ny /= n_len
                            nz /= n_len

                        # Two-sided diffuse lighting
                        diff = abs(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2])
                        intensity = ambient + (1.0 - ambient) * diff

                        # Apply lighting and clamp
                        r = min(1.0, r * intensity)
                        g = min(1.0, g * intensity)
                        b = min(1.0, b * intensity)

                        framebuffer[py, px, 0] = int(r * 255)
                        framebuffer[py, px, 1] = int(g * 255)
                        framebuffer[py, px, 2] = int(b * 255)
                        framebuffer[py, px, 3] = 255


@njit(cache=True)
def _draw_line_bresenham(
    framebuffer: np.ndarray,
    depth_buffer: np.ndarray,
    x0: int, y0: int, z0: float,
    x1: int, y1: int, z1: float,
    r0: float, g0: float, b0: float,
    r1: float, g1: float, b1: float,
    line_width: int,
) -> None:
    """Draw a single line using Bresenham's algorithm with no gaps."""
    height, width = framebuffer.shape[:2]
    half_width = line_width // 2

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Total length for interpolation
    total_steps = max(dx, dy)
    if total_steps == 0:
        total_steps = 1

    err = dx - dy
    step = 0

    while True:
        # Interpolation parameter
        t = step / total_steps if total_steps > 0 else 0.0

        # Interpolate depth and color
        z = z0 + t * (z1 - z0)
        r = r0 + t * (r1 - r0)
        g = g0 + t * (g1 - g0)
        b = b0 + t * (b1 - b0)

        # Draw pixel with line width
        for oy in range(-half_width, half_width + 1):
            for ox in range(-half_width, half_width + 1):
                px = x0 + ox
                py = y0 + oy

                if 0 <= px < width and 0 <= py < height:
                    if z < depth_buffer[py, px]:
                        depth_buffer[py, px] = z
                        framebuffer[py, px, 0] = int(min(1.0, r) * 255)
                        framebuffer[py, px, 1] = int(min(1.0, g) * 255)
                        framebuffer[py, px, 2] = int(min(1.0, b) * 255)
                        framebuffer[py, px, 3] = 255

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

        step += 1


@njit(parallel=True, cache=True)
def _rasterize_lines(
    framebuffer: np.ndarray,
    depth_buffer: np.ndarray,
    screen_verts: np.ndarray,
    colors: np.ndarray,
    segments: np.ndarray,
    line_width: int,
) -> None:
    """Rasterize line segments with depth testing using Bresenham's algorithm.

    Args:
        framebuffer: RGBA image buffer (H, W, 4) uint8
        depth_buffer: Depth buffer (H, W) float32
        screen_verts: Screen-space vertices (N, 4) [x, y, z, w]
        colors: Vertex colors (N, 3) float32
        segments: Line segment indices (S, 2) int32
        line_width: Line width in pixels
    """
    for si in prange(len(segments)):
        i0, i1 = segments[si]

        x0 = int(screen_verts[i0, 0])
        y0 = int(screen_verts[i0, 1])
        z0 = screen_verts[i0, 2]
        x1 = int(screen_verts[i1, 0])
        y1 = int(screen_verts[i1, 1])
        z1 = screen_verts[i1, 2]

        r0, g0, b0 = colors[i0, 0], colors[i0, 1], colors[i0, 2]
        r1, g1, b1 = colors[i1, 0], colors[i1, 1], colors[i1, 2]

        _draw_line_bresenham(
            framebuffer, depth_buffer,
            x0, y0, z0, x1, y1, z1,
            r0, g0, b0, r1, g1, b1,
            line_width
        )


def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from faces."""
    normals = np.zeros_like(vertices, dtype=np.float32)

    if len(faces) == 0:
        return normals

    # Compute face normals and accumulate to vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Accumulate face normals to vertices
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths > 1e-10, lengths, 1.0)
    normals = normals / lengths

    return normals


def _ortho_matrix(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    """Create orthographic projection matrix."""
    return np.array([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1],
    ], dtype=np.float32)


class SoftwareRenderer:
    """Software rasterizer for headless mesh rendering.

    Pure numpy/numba implementation that works on all platforms.
    """

    def __init__(self, args: Optional[dict] = None):
        if args is None:
            args = {}
        self._width = args.get("width", 640)
        self._height = args.get("height", 480)
        self._light_dir = np.array([0.3, 0.5, 0.8], dtype=np.float32)
        self._light_dir /= np.linalg.norm(self._light_dir)
        self._ambient = 0.3
        self._line_width = 1

    def render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        seg: np.ndarray,
        face: np.ndarray,
        output: str | None,
    ) -> Optional[np.ndarray]:
        """Render a mesh with vertex colors.

        Args:
            vert: Vertices (N, 3)
            color: Vertex colors (N, 3) or (N, 4)
            seg: Line segments (S, 2)
            face: Triangle faces (F, 3)
            output: Output file path (None to return array)

        Returns:
            RGBA image as numpy array if output is None
        """
        width, height = self._width, self._height

        # Initialize buffers
        framebuffer = np.ones((height, width, 4), dtype=np.uint8) * 255  # White background
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

        # Transform vertices: center, rotate, normalize
        vert = vert.copy().astype(np.float32)
        rad = np.radians(10)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)],
        ], dtype=np.float32)

        bounds = np.max(vert, axis=0) - np.min(vert, axis=0)
        center = (bounds / 2) + np.min(vert, axis=0)
        vert = vert - center
        vert = np.dot(vert, rotation_matrix.T)
        max_extent = np.max(np.abs(vert))
        if max_extent > 0:
            vert = vert / max_extent * 0.9

        # Prepare colors
        color = color.astype(np.float32)
        if color.shape[1] == 4:
            color = color[:, :3]

        # Setup projection matrix
        aspect = width / height
        if aspect >= 1:
            proj = _ortho_matrix(-aspect, aspect, -1, 1, -10, 10)
        else:
            proj = _ortho_matrix(-1, 1, -1/aspect, 1/aspect, -10, 10)

        # Transform to clip space
        ones = np.ones((len(vert), 1), dtype=np.float32)
        verts_h = np.hstack([vert, ones])  # (N, 4)
        clip_verts = np.dot(verts_h, proj.T)  # (N, 4)

        # Perspective divide (orthographic, so w=1)
        ndc_verts = clip_verts[:, :3] / clip_verts[:, 3:4]

        # NDC to screen coordinates
        screen_verts = np.zeros((len(vert), 4), dtype=np.float32)
        screen_verts[:, 0] = (ndc_verts[:, 0] + 1) * 0.5 * width   # x
        screen_verts[:, 1] = (1 - ndc_verts[:, 1]) * 0.5 * height  # y (flip Y)
        screen_verts[:, 2] = ndc_verts[:, 2]  # z for depth testing
        screen_verts[:, 3] = 1.0  # w

        # Render triangles if present
        if len(face) > 0:
            normals = _compute_normals(vert, face)
            face_arr = face.astype(np.int32)
            _rasterize_triangles(
                framebuffer, depth_buffer,
                screen_verts, color, normals, face_arr,
                self._light_dir, self._ambient
            )

        # Render lines if present
        if len(seg) > 0:
            seg_arr = seg.astype(np.int32)
            line_colors = color
            _rasterize_lines(
                framebuffer, depth_buffer,
                screen_verts, line_colors, seg_arr,
                self._line_width
            )

        # Output
        if output is not None:
            img = Image.fromarray(framebuffer, mode="RGBA")
            img.save(output)
            return None
        else:
            return framebuffer
