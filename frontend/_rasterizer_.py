# File: _rasterizer_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Pure software rasterizer backed by the Rust kernel.

Provides lightweight headless rendering without heavy dependencies.
"""

from typing import Optional

import numpy as np
from PIL import Image

from . import _rust  # type: ignore[attr-defined]

# Single source of truth for the renderer's default image dimensions.
# Both SoftwareRenderer and the CLI argparse defaults reference these, and
# the Rust render_default_args table (crates/ppf-cts-py/src/render_py.rs)
# must agree with them so the software and Mitsuba paths cannot diverge.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class SoftwareRenderer:
    """Software rasterizer for headless mesh rendering.

    Backed by a Rust kernel; works on all platforms without requiring a
    GPU or windowing system.
    """

    def __init__(self, args: Optional[dict] = None):
        """Initialize the renderer.

        Args:
            args (Optional[dict]): Optional configuration. Recognized keys are
                ``width`` (default :data:`DEFAULT_WIDTH`) and ``height``
                (default :data:`DEFAULT_HEIGHT`).
        """
        if args is None:
            args = {}
        self._width = args.get("width", DEFAULT_WIDTH)
        self._height = args.get("height", DEFAULT_HEIGHT)
        self._light_dir = np.array(
            _rust.normalize_light_dir(np.array([0.3, 0.5, 0.8], dtype=np.float32)),
            dtype=np.float32,
        )
        self._ambient = 0.3
        self._line_width = 1

    def render(
        self,
        vert: np.ndarray,
        color: np.ndarray,
        seg: np.ndarray,
        face: np.ndarray,
        output: Optional[str],
    ) -> Optional[np.ndarray]:
        """Render a mesh with vertex colors to a file or a numpy array.

        The mesh is centered, rotated slightly, and scaled to fit into the
        orthographic view volume before rasterization. The output is RGBA on a
        white background.

        Args:
            vert: Vertices (N, 3).
            color: Vertex colors (N, 3) or (N, 4); the alpha channel is dropped
                when four components are provided.
            seg: Line segment indices (S, 2).
            face: Triangle face indices (F, 3).
            output: Output file path, or ``None`` to return the image buffer.

        Returns:
            Optional[np.ndarray]: The RGBA image buffer (H, W, 4) uint8 when
            ``output`` is ``None``, otherwise ``None``.
        """
        width, height = self._width, self._height

        # Initialize buffers
        framebuffer = np.ones((height, width, 4), dtype=np.uint8) * 255  # White background
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

        # Run the centering + tilt + projection + screen-space pipeline in Rust.
        vert, screen_verts = _rust.render_transform(
            np.ascontiguousarray(vert, dtype=np.float32), int(width), int(height)
        )

        # Prepare colors
        color = color.astype(np.float32)
        if color.shape[1] == 4:
            color = color[:, :3]

        # Render triangles if present
        if len(face) > 0:
            face_arr = face.astype(np.int32)
            normals = _rust.normals(
                np.ascontiguousarray(vert, dtype=np.float32), face_arr
            )
            _rust.rasterize_triangles(
                framebuffer, depth_buffer,
                np.ascontiguousarray(screen_verts, dtype=np.float32),
                np.ascontiguousarray(color, dtype=np.float32),
                np.ascontiguousarray(normals, dtype=np.float32),
                face_arr,
                np.ascontiguousarray(self._light_dir, dtype=np.float32),
                float(self._ambient),
            )

        # Render lines if present
        if len(seg) > 0:
            seg_arr = seg.astype(np.int32)
            line_colors = color
            _rust.rasterize_lines(
                framebuffer, depth_buffer,
                np.ascontiguousarray(screen_verts, dtype=np.float32),
                np.ascontiguousarray(line_colors, dtype=np.float32),
                seg_arr,
                int(self._line_width),
            )

        # Output
        if output is not None:
            img = Image.fromarray(framebuffer, mode="RGBA")
            img.save(output)
            return None
        else:
            return framebuffer
