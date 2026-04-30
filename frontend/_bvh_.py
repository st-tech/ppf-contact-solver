# File: _bvh_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""AABB-based BVH for mesh queries using Numba with parallel ops.

Provides shared BVH infrastructure for triangles, edges, and points
used by both intersection and proximity detection modules.
"""

from collections.abc import Callable
from typing import Optional

import numpy as np

from numba import njit, prange


@njit(cache=True, inline="always")
def dot3(a: np.ndarray, b: np.ndarray) -> float:
    """Return the dot product of two 3D vectors without calling into BLAS."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _point_to_bbox_dist_sq(
    point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray
) -> float:
    """Return the squared distance from a point to an AABB (0 if inside)."""
    dist_sq = 0.0
    for i in range(3):
        if point[i] < bbox_min[i]:
            d = bbox_min[i] - point[i]
            dist_sq += d * d
        elif point[i] > bbox_max[i]:
            d = point[i] - bbox_max[i]
            dist_sq += d * d
    return dist_sq


@njit(cache=True)
def bbox_overlap(
    min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray
) -> bool:
    """Return True if the two AABBs overlap (touching counts as overlap)."""
    for i in range(3):  # noqa: SIM110
        if min1[i] > max2[i] or min2[i] > max1[i]:
            return False
    return True


@njit(parallel=True, cache=True)
def _flatten_tris_numba(
    verts: np.ndarray,
    tris: np.ndarray,
    centroids: np.ndarray,
    bboxes_min: np.ndarray,
    bboxes_max: np.ndarray,
):
    """Compute triangle centroids and bounding boxes in parallel."""
    n_tris = len(tris)
    for i in prange(n_tris):
        t0, t1, t2 = tris[i, 0], tris[i, 1], tris[i, 2]
        v0 = verts[t0]
        v1 = verts[t1]
        v2 = verts[t2]
        for d in range(3):
            centroids[i, d] = (v0[d] + v1[d] + v2[d]) / 3.0
            bboxes_min[i, d] = min(v0[d], min(v1[d], v2[d]))
            bboxes_max[i, d] = max(v0[d], max(v1[d], v2[d]))


@njit(parallel=True, cache=True)
def _flatten_edges_numba(
    verts: np.ndarray,
    edges: np.ndarray,
    centroids: np.ndarray,
    bboxes_min: np.ndarray,
    bboxes_max: np.ndarray,
):
    """Compute edge centroids and bounding boxes in parallel."""
    n_edges = len(edges)
    for i in prange(n_edges):
        e0, e1 = edges[i, 0], edges[i, 1]
        v0 = verts[e0]
        v1 = verts[e1]
        for d in range(3):
            centroids[i, d] = (v0[d] + v1[d]) / 2.0
            bboxes_min[i, d] = min(v0[d], v1[d])
            bboxes_max[i, d] = max(v0[d], v1[d])


@njit(parallel=True, cache=True)
def _compute_morton_codes(
    centroids: np.ndarray,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    morton_codes: np.ndarray,
):
    """Compute 30-bit Morton codes for centroids in parallel."""
    n = len(centroids)
    scale = scene_max - scene_min
    for d in range(3):
        if scale[d] < 1e-10:
            scale[d] = 1.0

    for i in prange(n):
        # Normalize to [0, 1]
        nx = (centroids[i, 0] - scene_min[0]) / scale[0]
        ny = (centroids[i, 1] - scene_min[1]) / scale[1]
        nz = (centroids[i, 2] - scene_min[2]) / scale[2]

        # Clamp to [0, 1]
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        nz = max(0.0, min(1.0, nz))

        # Convert to 10-bit integers
        ix = min(int(nx * 1023), 1023)
        iy = min(int(ny * 1023), 1023)
        iz = min(int(nz * 1023), 1023)

        # Interleave bits to form Morton code
        code = 0
        for bit in range(10):
            code |= ((ix >> bit) & 1) << (3 * bit)
            code |= ((iy >> bit) & 1) << (3 * bit + 1)
            code |= ((iz >> bit) & 1) << (3 * bit + 2)
        morton_codes[i] = code


@njit(cache=True)
def _build_bvh_structure(
    n_elems: int,
    max_leaf_size: int,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    node_depth: np.ndarray,
) -> int:
    """Build the BVH tree topology by midpoint splitting of Morton-sorted elements.

    Only populates the tree's child and leaf-range arrays; bounding
    boxes are filled in by later phases. Returns the total node count.
    """
    # Work stack: (node_idx, start, end, depth)
    stack = np.empty((128, 4), dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr, 0] = 0
    stack[stack_ptr, 1] = 0
    stack[stack_ptr, 2] = n_elems
    stack[stack_ptr, 3] = 0
    stack_ptr += 1

    node_count = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr, 0]
        start = stack[stack_ptr, 1]
        end = stack[stack_ptr, 2]
        depth = stack[stack_ptr, 3]
        count = end - start

        node_depth[node_idx] = depth

        if count <= max_leaf_size:
            node_elem_start[node_idx] = start
            node_elem_count[node_idx] = count
        else:
            mid = (start + end) // 2

            left_idx = node_count
            right_idx = node_count + 1
            node_count += 2

            node_left[node_idx] = left_idx
            node_right[node_idx] = right_idx

            stack[stack_ptr, 0] = right_idx
            stack[stack_ptr, 1] = mid
            stack[stack_ptr, 2] = end
            stack[stack_ptr, 3] = depth + 1
            stack_ptr += 1
            stack[stack_ptr, 0] = left_idx
            stack[stack_ptr, 1] = start
            stack[stack_ptr, 2] = mid
            stack[stack_ptr, 3] = depth + 1
            stack_ptr += 1

    return node_count


@njit(parallel=True, cache=True)
def _compute_leaf_bboxes(
    node_count: int,
    sorted_indices: np.ndarray,
    bboxes_min: np.ndarray,
    bboxes_max: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
):
    """Compute bounding boxes for leaf nodes in parallel."""
    for node_idx in prange(node_count):
        if node_elem_start[node_idx] >= 0:
            start = node_elem_start[node_idx]
            count = node_elem_count[node_idx]

            idx = sorted_indices[start]
            bmin_x = bboxes_min[idx, 0]
            bmin_y = bboxes_min[idx, 1]
            bmin_z = bboxes_min[idx, 2]
            bmax_x = bboxes_max[idx, 0]
            bmax_y = bboxes_max[idx, 1]
            bmax_z = bboxes_max[idx, 2]

            for i in range(1, count):
                idx = sorted_indices[start + i]
                if bboxes_min[idx, 0] < bmin_x:
                    bmin_x = bboxes_min[idx, 0]
                if bboxes_min[idx, 1] < bmin_y:
                    bmin_y = bboxes_min[idx, 1]
                if bboxes_min[idx, 2] < bmin_z:
                    bmin_z = bboxes_min[idx, 2]
                if bboxes_max[idx, 0] > bmax_x:
                    bmax_x = bboxes_max[idx, 0]
                if bboxes_max[idx, 1] > bmax_y:
                    bmax_y = bboxes_max[idx, 1]
                if bboxes_max[idx, 2] > bmax_z:
                    bmax_z = bboxes_max[idx, 2]

            node_bbox_min[node_idx, 0] = bmin_x
            node_bbox_min[node_idx, 1] = bmin_y
            node_bbox_min[node_idx, 2] = bmin_z
            node_bbox_max[node_idx, 0] = bmax_x
            node_bbox_max[node_idx, 1] = bmax_y
            node_bbox_max[node_idx, 2] = bmax_z


@njit(cache=True)
def _propagate_bboxes_bottom_up(
    node_count: int,
    max_depth: int,
    node_depth: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
):
    """Propagate bounding boxes from leaves to root."""
    for depth in range(max_depth - 1, -1, -1):
        for node_idx in range(node_count):
            if node_depth[node_idx] == depth and node_left[node_idx] >= 0:
                left = node_left[node_idx]
                right = node_right[node_idx]
                for d in range(3):
                    node_bbox_min[node_idx, d] = min(
                        node_bbox_min[left, d], node_bbox_min[right, d]
                    )
                    node_bbox_max[node_idx, d] = max(
                        node_bbox_max[left, d], node_bbox_max[right, d]
                    )


def _build_bvh_flat(
    centroids: np.ndarray,
    bboxes_min: np.ndarray,
    bboxes_max: np.ndarray,
    max_leaf_size: int = 8,
):
    """Build a flat BVH over elements ordered by 30-bit Morton codes.

    Returns a tuple of arrays describing the node layout plus the
    Morton-sorted element indices: (node_bbox_min, node_bbox_max,
    node_left, node_right, node_elem_start, node_elem_count,
    sorted_indices). Leaf nodes have node_elem_start >= 0; internal
    nodes have node_elem_start == -1 and valid node_left/node_right.
    """
    n_elems = len(centroids)
    if n_elems == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )

    # Compute scene bounds
    scene_min = centroids.min(axis=0)
    scene_max = centroids.max(axis=0)

    # Compute Morton codes in parallel
    morton_codes = np.empty(n_elems, dtype=np.int32)
    _compute_morton_codes(centroids, scene_min, scene_max, morton_codes)

    # Sort by Morton codes
    sorted_indices = np.argsort(morton_codes).astype(np.int32)

    # Allocate node arrays
    max_nodes = 2 * n_elems
    node_bbox_min = np.zeros((max_nodes, 3), dtype=np.float64)
    node_bbox_max = np.zeros((max_nodes, 3), dtype=np.float64)
    node_left = np.full(max_nodes, -1, dtype=np.int32)
    node_right = np.full(max_nodes, -1, dtype=np.int32)
    node_elem_start = np.full(max_nodes, -1, dtype=np.int32)
    node_elem_count = np.zeros(max_nodes, dtype=np.int32)
    node_depth = np.zeros(max_nodes, dtype=np.int32)

    # Phase 1: Build tree structure (fast, no bbox computation)
    node_count = _build_bvh_structure(
        n_elems,
        max_leaf_size,
        node_left,
        node_right,
        node_elem_start,
        node_elem_count,
        node_depth,
    )

    # Phase 2: Compute leaf bounding boxes in parallel
    _compute_leaf_bboxes(
        node_count,
        sorted_indices,
        bboxes_min,
        bboxes_max,
        node_elem_start,
        node_elem_count,
        node_bbox_min,
        node_bbox_max,
    )

    # Phase 3: Propagate bboxes bottom-up
    max_depth = int(node_depth[:node_count].max()) + 1
    _propagate_bboxes_bottom_up(
        node_count,
        max_depth,
        node_depth,
        node_left,
        node_right,
        node_bbox_min,
        node_bbox_max,
    )

    return (
        node_bbox_min[:node_count].copy(),
        node_bbox_max[:node_count].copy(),
        node_left[:node_count].copy(),
        node_right[:node_count].copy(),
        node_elem_start[:node_count].copy(),
        node_elem_count[:node_count].copy(),
        sorted_indices,
    )


@njit(cache=True)
def _query_bvh_single(
    point: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_tri_start: np.ndarray,
    node_tri_count: np.ndarray,
    tri_indices_flat: np.ndarray,
):
    """Query BVH for the closest triangle to a point.

    Iterative traversal with an explicit node-index stack; the nearer child
    is pushed last so it is popped first. Returns (best_dist_sq, best_tri,
    best_bary) where best_bary are the barycentric coordinates of the
    closest point on best_tri.
    """
    best_dist_sq = np.inf
    best_tri = 0
    best_bary = np.array([1.0, 0.0, 0.0])

    # Stack of node indices for iterative traversal
    stack = np.zeros(64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0  # Start with root node
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        # Early exit if this node's bbox is farther than current best
        bbox_dist_sq = _point_to_bbox_dist_sq(
            point, node_bbox_min[node_idx], node_bbox_max[node_idx]
        )
        if bbox_dist_sq >= best_dist_sq:
            continue

        if node_tri_start[node_idx] >= 0:
            # Leaf node: check all triangles
            start = node_tri_start[node_idx]
            count = node_tri_count[node_idx]
            for i in range(count):
                ti = tri_indices_flat[start + i]
                a = verts[tris[ti, 0]]
                b = verts[tris[ti, 1]]
                c = verts[tris[ti, 2]]
                closest, bary = closest_point_on_triangle(point, a, b, c)
                diff = point - closest
                dist_sq = dot3(diff, diff)
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_tri = ti
                    best_bary = bary
        else:
            # Internal node: push children (closer one last so it's popped first)
            left_idx = node_left[node_idx]
            right_idx = node_right[node_idx]

            left_dist = _point_to_bbox_dist_sq(
                point, node_bbox_min[left_idx], node_bbox_max[left_idx]
            )
            right_dist = _point_to_bbox_dist_sq(
                point, node_bbox_min[right_idx], node_bbox_max[right_idx]
            )

            # Push farther child first, then closer (so closer is processed first)
            if left_dist < right_dist:
                stack[stack_ptr] = right_idx
                stack_ptr += 1
                stack[stack_ptr] = left_idx
                stack_ptr += 1
            else:
                stack[stack_ptr] = left_idx
                stack_ptr += 1
                stack[stack_ptr] = right_idx
                stack_ptr += 1

    return best_dist_sq, best_tri, best_bary


# Triangle is considered degenerate when ||b1 x b2|| falls below this.
# Chosen well below any legitimate fTetWild triangle area so we only fall
# back in truly pathological cases.
_FRAME_DEGEN_EPS = 1e-20


@njit(parallel=True, cache=True)
def _query_and_build_frame_parallel(
    points: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_tri_start: np.ndarray,
    node_tri_count: np.ndarray,
    tri_indices_flat: np.ndarray,
    out_tri_indices: np.ndarray,
    out_coefs: np.ndarray,
):
    """Find closest triangle per point and solve c = B0^-1 (p - x0).

    Frame B0 = [b1 | b2 | b3] where b1 = x1-x0, b2 = x2-x0, and b3 is
    the unit normal (b1 x b2)/||b1 x b2||. Because b3 is orthogonal to
    b1 and b2, the 3x3 solve decouples:
      c3 = (p - x0) . b3
      [c1, c2] = ((b1, b2) Gram)^-1 [(p - x0).b1, (p - x0).b2]
    with Gram determinant = ||b1 x b2||^2. For degenerate triangles
    (||b1 x b2||^2 below _FRAME_DEGEN_EPS) the coefficients are set to
    zero.
    """
    n_points = len(points)
    for i in prange(n_points):
        _, best_tri, _ = _query_bvh_single(
            points[i],
            verts,
            tris,
            node_bbox_min,
            node_bbox_max,
            node_left,
            node_right,
            node_tri_start,
            node_tri_count,
            tri_indices_flat,
        )
        out_tri_indices[i] = best_tri

        t0 = tris[best_tri, 0]
        t1 = tris[best_tri, 1]
        t2 = tris[best_tri, 2]
        x0x = verts[t0, 0]
        x0y = verts[t0, 1]
        x0z = verts[t0, 2]
        b1x = verts[t1, 0] - x0x
        b1y = verts[t1, 1] - x0y
        b1z = verts[t1, 2] - x0z
        b2x = verts[t2, 0] - x0x
        b2y = verts[t2, 1] - x0y
        b2z = verts[t2, 2] - x0z

        nx = b1y * b2z - b1z * b2y
        ny = b1z * b2x - b1x * b2z
        nz = b1x * b2y - b1y * b2x
        n_sq = nx * nx + ny * ny + nz * nz

        if n_sq < _FRAME_DEGEN_EPS:
            out_coefs[i, 0] = 0.0
            out_coefs[i, 1] = 0.0
            out_coefs[i, 2] = 0.0
            continue

        dx = points[i, 0] - x0x
        dy = points[i, 1] - x0y
        dz = points[i, 2] - x0z

        db1 = dx * b1x + dy * b1y + dz * b1z
        db2 = dx * b2x + dy * b2y + dz * b2z
        b1b1 = b1x * b1x + b1y * b1y + b1z * b1z
        b2b2 = b2x * b2x + b2y * b2y + b2z * b2z
        b1b2 = b1x * b2x + b1y * b2y + b1z * b2z

        out_coefs[i, 0] = (b2b2 * db1 - b1b2 * db2) / n_sq
        out_coefs[i, 1] = (b1b1 * db2 - b1b2 * db1) / n_sq
        # c3 = (p - x0) . n̂ = (p - x0) . n / ||n||
        inv_nlen = 1.0 / np.sqrt(n_sq)
        out_coefs[i, 2] = (dx * nx + dy * ny + dz * nz) * inv_nlen


def compute_frame_mapping(
    orig_vert: np.ndarray,
    new_vert: np.ndarray,
    new_tri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed each original vertex in the local frame of its closest new-mesh triangle.

    For each point p in orig_vert, find the closest triangle T on
    (new_vert, new_tri) and store coefficients c = B0^-1 (p - x0), where
    B0 = [b1 | b2 | b3]:
        b1 = x1 - x0,  b2 = x2 - x0,  b3 = (b1 x b2) / ||b1 x b2||.
    c1, c2 are affine coords in the edge basis (they scale with the
    triangle) and c3 is a signed offset along the unit normal (in absolute
    world units). At runtime p' = x0' + c1*(x1' - x0') + c2*(x2' - x0') +
    c3 * n_hat' preserves the original normal offset regardless of
    triangle scaling, avoiding the shrink artifact of a pure barycentric
    projection.

    Args:
        orig_vert: Points to embed (N, 3).
        new_vert: Target mesh vertices (P, 3).
        new_tri: Target mesh triangles (Q, 3).

    Returns:
        tri_indices: Closest triangle index per input point (N,).
        coefs: Frame coefficients (c1, c2, c3) per input point (N, 3).
    """
    n_orig = len(orig_vert)

    orig_vert = np.ascontiguousarray(orig_vert, dtype=np.float64)
    new_vert = np.ascontiguousarray(new_vert, dtype=np.float64)
    new_tri = np.ascontiguousarray(new_tri, dtype=np.int32)

    v0 = new_vert[new_tri[:, 0]]
    v1 = new_vert[new_tri[:, 1]]
    v2 = new_vert[new_tri[:, 2]]

    tri_verts = np.stack([v0, v1, v2], axis=1)
    tri_bboxes_min = tri_verts.min(axis=1)
    tri_bboxes_max = tri_verts.max(axis=1)
    tri_centroids = tri_verts.mean(axis=1)

    (
        node_bbox_min,
        node_bbox_max,
        node_left,
        node_right,
        node_tri_start,
        node_tri_count,
        tri_indices_flat,
    ) = _build_bvh_flat(tri_centroids, tri_bboxes_min, tri_bboxes_max)

    tri_indices = np.zeros(n_orig, dtype=np.int32)
    coefs = np.zeros((n_orig, 3), dtype=np.float64)

    _query_and_build_frame_parallel(
        orig_vert,
        new_vert,
        new_tri,
        node_bbox_min,
        node_bbox_max,
        node_left,
        node_right,
        node_tri_start,
        node_tri_count,
        tri_indices_flat,
        tri_indices,
        coefs,
    )

    return tri_indices, coefs


@njit(parallel=True, cache=True)
def _interpolate_surface_parallel(
    deformed_vert: np.ndarray,
    surf_tri: np.ndarray,
    tri_indices: np.ndarray,
    coefs: np.ndarray,
    out: np.ndarray,
):
    """Reconstruct positions from frame coefs: p' = x0' + c1*b1' + c2*b2' + c3*n̂'."""
    n = len(tri_indices)
    for i in prange(n):
        ti = tri_indices[i]
        t0 = surf_tri[ti, 0]
        t1 = surf_tri[ti, 1]
        t2 = surf_tri[ti, 2]
        c0 = coefs[i, 0]
        c1 = coefs[i, 1]
        c2 = coefs[i, 2]

        x0x = deformed_vert[t0, 0]
        x0y = deformed_vert[t0, 1]
        x0z = deformed_vert[t0, 2]
        b1x = deformed_vert[t1, 0] - x0x
        b1y = deformed_vert[t1, 1] - x0y
        b1z = deformed_vert[t1, 2] - x0z
        b2x = deformed_vert[t2, 0] - x0x
        b2y = deformed_vert[t2, 1] - x0y
        b2z = deformed_vert[t2, 2] - x0z

        nx = b1y * b2z - b1z * b2y
        ny = b1z * b2x - b1x * b2z
        nz = b1x * b2y - b1y * b2x
        n_sq = nx * nx + ny * ny + nz * nz

        if n_sq < _FRAME_DEGEN_EPS:
            # Degenerate deformed triangle: drop the normal term rather than
            # emit NaN. c3 was stored relative to the rest triangle's normal
            # direction, which no longer exists.
            out[i, 0] = x0x + c0 * b1x + c1 * b2x
            out[i, 1] = x0y + c0 * b1y + c1 * b2y
            out[i, 2] = x0z + c0 * b1z + c1 * b2z
        else:
            inv_nlen = 1.0 / np.sqrt(n_sq)
            out[i, 0] = x0x + c0 * b1x + c1 * b2x + c2 * nx * inv_nlen
            out[i, 1] = x0y + c0 * b1y + c1 * b2y + c2 * ny * inv_nlen
            out[i, 2] = x0z + c0 * b1z + c1 * b2z + c2 * nz * inv_nlen


def interpolate_surface(
    deformed_vert: np.ndarray,
    surf_tri: np.ndarray,
    tri_indices: np.ndarray,
    coefs: np.ndarray,
) -> np.ndarray:
    """Reconstruct embedded point positions from a deformed host mesh.

    Applies the frame-embedding reconstruction p' = x0' + c1*b1' + c2*b2'
    + c3*n_hat' per input point using the mapping produced by
    compute_frame_mapping. Parallelized with Numba.

    Args:
        deformed_vert: Deformed host mesh vertices (P, 3).
        surf_tri: Host mesh triangles (Q, 3).
        tri_indices: Closest triangle index per embedded point (N,).
        coefs: Frame coefficients (c1, c2, c3) per embedded point (N, 3).

    Returns:
        Reconstructed point positions (N, 3).
    """
    n_orig = len(tri_indices)

    deformed_vert = np.ascontiguousarray(deformed_vert, dtype=np.float64)
    surf_tri = np.ascontiguousarray(surf_tri, dtype=np.int32)
    tri_indices = np.ascontiguousarray(tri_indices, dtype=np.int32)
    coefs = np.ascontiguousarray(coefs, dtype=np.float64)

    out = np.zeros((n_orig, 3), dtype=np.float64)
    _interpolate_surface_parallel(
        deformed_vert, surf_tri, tri_indices, coefs, out
    )
    return out


# =============================================================================
# Edge extraction utilities
# =============================================================================


def extract_unique_edges(F: np.ndarray) -> np.ndarray:
    """Extract unique undirected edges from triangle faces.

    Args:
        F: Triangle faces (M, 3).

    Returns:
        Unique edges (K, 2) with the two vertex indices sorted per edge.
    """
    if len(F) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # Generate all edges from triangles
    edges_list = []
    for i in range(len(F)):
        t = F[i]
        edges_list.append((min(t[0], t[1]), max(t[0], t[1])))
        edges_list.append((min(t[1], t[2]), max(t[1], t[2])))
        edges_list.append((min(t[2], t[0]), max(t[2], t[0])))

    # Get unique edges
    unique_edges = list(set(edges_list))
    return np.array(unique_edges, dtype=np.int32)


@njit(parallel=True, cache=True)
def _generate_edge_keys(
    F: np.ndarray,
    edge_keys: np.ndarray,
    tri_indices: np.ndarray,
    max_vert: int,
):
    """Generate edge keys and triangle indices in parallel."""
    n_tris = len(F)
    for ti in prange(n_tris):
        t0, t1, t2 = F[ti, 0], F[ti, 1], F[ti, 2]
        # Edge 0: t0-t1
        if t0 < t1:
            edge_keys[ti * 3] = t0 * max_vert + t1
        else:
            edge_keys[ti * 3] = t1 * max_vert + t0
        tri_indices[ti * 3] = ti
        # Edge 1: t1-t2
        if t1 < t2:
            edge_keys[ti * 3 + 1] = t1 * max_vert + t2
        else:
            edge_keys[ti * 3 + 1] = t2 * max_vert + t1
        tri_indices[ti * 3 + 1] = ti
        # Edge 2: t2-t0
        if t2 < t0:
            edge_keys[ti * 3 + 2] = t2 * max_vert + t0
        else:
            edge_keys[ti * 3 + 2] = t0 * max_vert + t2
        tri_indices[ti * 3 + 2] = ti


@njit(cache=True)
def _collect_unique_edges_from_keys(
    sorted_keys: np.ndarray,
    sorted_tri_indices: np.ndarray,
    out_edges: np.ndarray,
    out_edge_to_tri: np.ndarray,
    max_vert: int,
) -> int:
    """Collect unique edges and their parent triangles from sorted keys."""
    n = len(sorted_keys)
    if n == 0:
        return 0

    # First edge
    key = sorted_keys[0]
    out_edges[0, 0] = key // max_vert
    out_edges[0, 1] = key % max_vert
    out_edge_to_tri[0, 0] = sorted_tri_indices[0]
    out_edge_to_tri[0, 1] = -1
    count = 1
    prev_key = key

    for i in range(1, n):
        key = sorted_keys[i]
        if key == prev_key:
            # Same edge, add second triangle
            if out_edge_to_tri[count - 1, 1] < 0:
                out_edge_to_tri[count - 1, 1] = sorted_tri_indices[i]
        else:
            # New edge
            out_edges[count, 0] = key // max_vert
            out_edges[count, 1] = key % max_vert
            out_edge_to_tri[count, 0] = sorted_tri_indices[i]
            out_edge_to_tri[count, 1] = -1
            count += 1
            prev_key = key

    return count


def extract_edges_with_tri_map(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique edges and map each edge to its parent triangle(s).

    Args:
        F: Triangle faces (M, 3).

    Returns:
        edges: Unique edges (K, 2).
        edge_to_tri: For each edge, up to two indices of triangles
            containing it (K, 2). The second index is -1 for boundary
            edges that belong to only one triangle. For non-manifold
            edges shared by three or more triangles, only the first two
            encountered are recorded.
    """
    if len(F) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)

    n_tris = len(F)
    n_all_edges = n_tris * 3
    max_vert = int(np.max(F)) + 1

    # Generate edge keys in parallel
    edge_keys = np.empty(n_all_edges, dtype=np.int64)
    tri_indices = np.empty(n_all_edges, dtype=np.int32)
    _generate_edge_keys(F, edge_keys, tri_indices, max_vert)

    # Sort by edge keys
    sort_idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_idx]
    sorted_tri_indices = tri_indices[sort_idx]

    # Collect unique edges
    out_edges = np.empty((n_all_edges, 2), dtype=np.int32)
    out_edge_to_tri = np.empty((n_all_edges, 2), dtype=np.int32)
    n_unique = _collect_unique_edges_from_keys(
        sorted_keys, sorted_tri_indices, out_edges, out_edge_to_tri, max_vert
    )

    return out_edges[:n_unique].copy(), out_edge_to_tri[:n_unique].copy()


# =============================================================================
# BVH data structure for multiple element types
# =============================================================================


class ElementBVH:
    """Container holding the flat BVH arrays for a single element type."""

    def __init__(
        self,
        node_bbox_min: np.ndarray,
        node_bbox_max: np.ndarray,
        node_left: np.ndarray,
        node_right: np.ndarray,
        node_elem_start: np.ndarray,
        node_elem_count: np.ndarray,
        elem_indices_flat: np.ndarray,
        elem_bboxes_min: np.ndarray,
        elem_bboxes_max: np.ndarray,
    ):
        self.node_bbox_min = node_bbox_min
        self.node_bbox_max = node_bbox_max
        self.node_left = node_left
        self.node_right = node_right
        self.node_elem_start = node_elem_start
        self.node_elem_count = node_elem_count
        self.elem_indices_flat = elem_indices_flat
        self.elem_bboxes_min = elem_bboxes_min
        self.elem_bboxes_max = elem_bboxes_max


class MeshBVH:
    """Container holding BVHs for triangles, edges, and points."""

    def __init__(
        self,
        verts: np.ndarray,
        tris: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        edge_to_tri: Optional[np.ndarray] = None,
        on_flatten: Optional[Callable[[], None]] = None,
        on_build: Optional[Callable[[], None]] = None,
    ):
        """Build BVHs for the triangle, edge, and point elements of a mesh.

        Triangle, edge, and point BVHs are each built only if the
        corresponding element set is non-empty. When tris is provided but
        edges is None, both edges and edge_to_tri are derived from tris
        via extract_edges_with_tri_map.

        Args:
            verts: Vertices (N, 3).
            tris: Triangle faces (M, 3), optional.
            edges: Edge segments (K, 2), optional. If None and tris is
                provided, edges are extracted from tris.
            edge_to_tri: Mapping from edges to parent triangles (K, 2),
                optional. Ignored if edges is None.
            on_flatten: Callback invoked after the flatten phase completes.
            on_build: Callback invoked after the build phase completes.
        """
        self.verts = np.ascontiguousarray(verts, dtype=np.float64)
        self.n_verts = len(verts)

        # Handle triangles
        if tris is not None:
            self.tris = np.ascontiguousarray(tris, dtype=np.int32)
            self.n_tris = len(tris)
        else:
            self.tris = np.zeros((0, 3), dtype=np.int32)
            self.n_tris = 0

        # Handle edges
        if edges is not None:
            self.edges = np.ascontiguousarray(edges, dtype=np.int32)
            self.edge_to_tri = edge_to_tri
        elif self.n_tris > 0:
            self.edges, self.edge_to_tri = extract_edges_with_tri_map(self.tris)
        else:
            self.edges = np.zeros((0, 2), dtype=np.int32)
            self.edge_to_tri = None
        self.n_edges = len(self.edges)

        # Flatten arrays for all element types
        tri_data: Optional[tuple] = None
        edge_data: Optional[tuple] = None
        point_data: Optional[tuple] = None

        if self.n_tris > 0:
            tri_data = self._flatten_tri()
        if self.n_edges > 0:
            edge_data = self._flatten_edge()
        if self.n_verts > 0:
            point_data = self._flatten_point()

        if on_flatten is not None:
            on_flatten()

        # Build the three BVHs (tri / edge / point). We used to dispatch
        # them across a ``ThreadPoolExecutor(max_workers=3)``, but each
        # ``_build_bvh_from_data`` calls ``parallel=True`` numba kernels
        # (``_compute_morton_codes``, ``_compute_leaf_bboxes``); having
        # three Python threads concurrently invoke parallel-njit code is
        # exactly the access pattern numba's workqueue threading layer
        # aborts on with "Concurrent access has been detected" (it's
        # safe under TBB, but TBB has no macOS arm64 wheel).
        #
        # Running sequentially is the right fix: the inner numba code
        # is already parallel, so the outer Python-level fan-out only
        # adds at most 3x speedup on a thread-pool that contends with
        # the inner workers anyway. Cleaner state, no platform-conditional
        # threading layer, no random aborts.
        self.tri_bvh: Optional[ElementBVH] = None
        self.edge_bvh: Optional[ElementBVH] = None
        self.point_bvh: Optional[ElementBVH] = None

        if tri_data is not None:
            self.tri_bvh = self._build_bvh_from_data(*tri_data)
        if edge_data is not None:
            self.edge_bvh = self._build_bvh_from_data(*edge_data)
        if point_data is not None:
            self.point_bvh = self._build_bvh_from_data(*point_data)

        if on_build is not None:
            on_build()

    def _flatten_tri(self) -> tuple:
        """Flatten triangle data for BVH building."""
        n = self.n_tris
        centroids = np.empty((n, 3), dtype=np.float64)
        bboxes_min = np.empty((n, 3), dtype=np.float64)
        bboxes_max = np.empty((n, 3), dtype=np.float64)
        _flatten_tris_numba(self.verts, self.tris, centroids, bboxes_min, bboxes_max)
        return centroids, bboxes_min, bboxes_max

    def _flatten_edge(self) -> tuple:
        """Flatten edge data for BVH building."""
        n = self.n_edges
        centroids = np.empty((n, 3), dtype=np.float64)
        bboxes_min = np.empty((n, 3), dtype=np.float64)
        bboxes_max = np.empty((n, 3), dtype=np.float64)
        _flatten_edges_numba(self.verts, self.edges, centroids, bboxes_min, bboxes_max)
        return centroids, bboxes_min, bboxes_max

    def _flatten_point(self) -> tuple:
        """Flatten point data for BVH building."""
        bboxes_min = self.verts.copy()
        bboxes_max = self.verts.copy()
        centroids = self.verts.copy()
        return centroids, bboxes_min, bboxes_max

    def _build_bvh_from_data(
        self, centroids: np.ndarray, bboxes_min: np.ndarray, bboxes_max: np.ndarray
    ) -> ElementBVH:
        """Build an ElementBVH from pre-flattened centroids and per-element bboxes."""
        bvh_data = _build_bvh_flat(centroids, bboxes_min, bboxes_max)
        return ElementBVH(
            bvh_data[0],
            bvh_data[1],
            bvh_data[2],
            bvh_data[3],
            bvh_data[4],
            bvh_data[5],
            bvh_data[6],
            bboxes_min,
            bboxes_max,
        )


# Deferred import to avoid circular dependency
# (proximity imports from bvh, bvh needs this function from proximity)
from ._proximity_ import closest_point_on_triangle  # noqa: E402
