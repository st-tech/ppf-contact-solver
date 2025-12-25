# File: _bvh_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""AABB-based BVH for mesh queries using Numba with parallel ops.

Provides shared BVH infrastructure for triangles, edges, and points
used by both intersection and proximity detection modules.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from numba import njit, prange


@njit(cache=True, inline="always")
def _dot3(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 3D vectors. Avoids scipy BLAS dependency."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _closest_point_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
):
    """Find closest point on triangle ABC to point P. Returns (closest_point, bary_coords)."""
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = _dot3(ab, ap)
    d2 = _dot3(ac, ap)

    # Check if P is in vertex region outside A
    if d1 <= 0 and d2 <= 0:
        return a.copy(), np.array([1.0, 0.0, 0.0])

    bp = p - b
    d3 = _dot3(ab, bp)
    d4 = _dot3(ac, bp)

    # Check if P is in vertex region outside B
    if d3 >= 0 and d4 <= d3:
        return b.copy(), np.array([0.0, 1.0, 0.0])

    # Check if P is in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        denom = d1 - d3
        v = d1 / denom if denom != 0 else 0.0
        return a + v * ab, np.array([1.0 - v, v, 0.0])

    cp = p - c
    d5 = _dot3(ab, cp)
    d6 = _dot3(ac, cp)

    # Check if P is in vertex region outside C
    if d6 >= 0 and d5 <= d6:
        return c.copy(), np.array([0.0, 0.0, 1.0])

    # Check if P is in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        denom = d2 - d6
        w = d2 / denom if denom != 0 else 0.0
        return a + w * ac, np.array([1.0 - w, 0.0, w])

    # Check if P is in edge region of BC
    va = d3 * d6 - d5 * d4
    d4_d3 = d4 - d3
    d5_d6 = d5 - d6
    if va <= 0 and d4_d3 >= 0 and d5_d6 >= 0:
        denom = d4_d3 + d5_d6
        w = d4_d3 / denom if denom != 0 else 0.0
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])

    # P is inside the triangle
    denom = va + vb + vc
    if denom == 0:
        return a.copy(), np.array([1.0, 0.0, 0.0])
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w
    return a + v * ab + w * ac, np.array([u, v, w])


@njit(cache=True)
def _point_to_bbox_dist_sq(
    point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray
) -> float:
    """Compute squared distance from point to AABB."""
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
    """Check if two AABBs overlap."""
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
    """Build BVH tree structure only (no bbox computation)."""
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
    """Build a flattened BVH structure using Morton code ordering."""
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
    """Query BVH for closest point using iterative traversal with stack."""
    best_dist_sq = np.inf
    best_tri = 0
    best_bary = np.array([1.0, 0.0, 0.0])

    # Stack for iterative traversal (node_idx, skip flag)
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
                closest, bary = _closest_point_on_triangle(point, a, b, c)
                diff = point - closest
                dist_sq = _dot3(diff, diff)
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


@njit(parallel=True, cache=True)
def _query_bvh_parallel(
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
    out_bary_coords: np.ndarray,
):
    """Query BVH for multiple points in parallel."""
    n_points = len(points)
    for i in prange(n_points):
        _, best_tri, best_bary = _query_bvh_single(
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
        out_bary_coords[i, 0] = best_bary[0]
        out_bary_coords[i, 1] = best_bary[1]
        out_bary_coords[i, 2] = best_bary[2]


def compute_barycentric_mapping(
    orig_vert: np.ndarray,
    new_vert: np.ndarray,
    new_tri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute barycentric mapping from new surface mesh to original surface vertices.

    For each original surface vertex, find the closest point on the new surface mesh
    and compute its barycentric coordinates within the containing triangle.
    Uses a custom AABB-BVH with Numba JIT and parallel processing for performance.

    Args:
        orig_vert: Original surface vertices (N, 3)
        new_vert: New surface vertices from tetrahedralization (P, 3)
        new_tri: New surface triangles from tetrahedralization (Q, 3)

    Returns:
        tri_indices: Triangle index in new mesh for each original vertex (N,)
        bary_coords: Barycentric coordinates for each original vertex (N, 3)
    """
    n_orig = len(orig_vert)

    # Ensure contiguous float64 arrays
    orig_vert = np.ascontiguousarray(orig_vert, dtype=np.float64)
    new_vert = np.ascontiguousarray(new_vert, dtype=np.float64)
    new_tri = np.ascontiguousarray(new_tri, dtype=np.int32)

    # Compute triangle bounding boxes and centroids
    v0 = new_vert[new_tri[:, 0]]
    v1 = new_vert[new_tri[:, 1]]
    v2 = new_vert[new_tri[:, 2]]

    tri_verts = np.stack([v0, v1, v2], axis=1)  # (n_tris, 3, 3)
    tri_bboxes_min = tri_verts.min(axis=1)
    tri_bboxes_max = tri_verts.max(axis=1)
    tri_centroids = tri_verts.mean(axis=1)

    # Build flattened BVH
    (
        node_bbox_min,
        node_bbox_max,
        node_left,
        node_right,
        node_tri_start,
        node_tri_count,
        tri_indices_flat,
    ) = _build_bvh_flat(tri_centroids, tri_bboxes_min, tri_bboxes_max)

    # Output arrays
    tri_indices = np.zeros(n_orig, dtype=np.int32)
    bary_coords = np.zeros((n_orig, 3), dtype=np.float64)

    # Query all points in parallel
    _query_bvh_parallel(
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
        bary_coords,
    )

    return tri_indices, bary_coords


@njit(parallel=True, cache=True)
def _interpolate_surface_parallel(
    deformed_vert: np.ndarray,
    surf_tri: np.ndarray,
    tri_indices: np.ndarray,
    bary_coords: np.ndarray,
    out: np.ndarray,
):
    """Interpolate surface positions in parallel."""
    n = len(tri_indices)
    for i in prange(n):
        ti = tri_indices[i]
        t0, t1, t2 = surf_tri[ti, 0], surf_tri[ti, 1], surf_tri[ti, 2]
        b0, b1, b2 = bary_coords[i, 0], bary_coords[i, 1], bary_coords[i, 2]
        for j in range(3):
            out[i, j] = (
                b0 * deformed_vert[t0, j]
                + b1 * deformed_vert[t1, j]
                + b2 * deformed_vert[t2, j]
            )


def interpolate_surface(
    deformed_vert: np.ndarray,
    surf_tri: np.ndarray,
    tri_indices: np.ndarray,
    bary_coords: np.ndarray,
) -> np.ndarray:
    """Interpolate deformed tet mesh vertices back to original surface vertices.

    Uses the stored barycentric mapping to compute positions of original surface
    vertices based on the deformed tet mesh surface. Parallelized with Numba.

    Args:
        deformed_vert: Deformed tet mesh vertices (P, 3)
        surf_tri: Surface triangles of tet mesh (Q, 3)
        tri_indices: Triangle index for each original vertex (N,)
        bary_coords: Barycentric coordinates for each original vertex (N, 3)

    Returns:
        Interpolated vertex positions matching original surface vertex count (N, 3)
    """
    n_orig = len(tri_indices)

    # Ensure contiguous arrays
    deformed_vert = np.ascontiguousarray(deformed_vert, dtype=np.float64)
    surf_tri = np.ascontiguousarray(surf_tri, dtype=np.int32)
    tri_indices = np.ascontiguousarray(tri_indices, dtype=np.int32)
    bary_coords = np.ascontiguousarray(bary_coords, dtype=np.float64)

    out = np.zeros((n_orig, 3), dtype=np.float64)
    _interpolate_surface_parallel(
        deformed_vert, surf_tri, tri_indices, bary_coords, out
    )
    return out


# =============================================================================
# Shared primitive tests for intersection and proximity
# =============================================================================


@njit(cache=True)
def point_point_dist_sq(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute squared distance between two points."""
    d = p1 - p2
    return _dot3(d, d)


@njit(cache=True)
def point_edge_dist_sq(p: np.ndarray, e0: np.ndarray, e1: np.ndarray) -> float:
    """Compute squared distance from point to edge segment."""
    edge = e1 - e0
    edge_len_sq = _dot3(edge, edge)
    if edge_len_sq < 1e-14:
        d = p - e0
        return _dot3(d, d)

    t = _dot3(p - e0, edge) / edge_len_sq
    t = max(0.0, min(1.0, t))
    closest = e0 + t * edge
    d = p - closest
    return _dot3(d, d)


@njit(cache=True)
def edge_edge_dist_sq(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> float:
    """Compute squared distance between two edge segments."""
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0

    a = _dot3(d1, d1)
    e = _dot3(d2, d2)
    f = _dot3(d2, r)

    eps = 1e-14

    if a < eps and e < eps:
        d = a0 - b0
        return _dot3(d, d)

    if a < eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    elif e < eps:
        t = 0.0
        s = max(0.0, min(1.0, -_dot3(d1, r) / a))
    else:
        b_val = _dot3(d1, d2)
        c = _dot3(d1, r)
        denom = a * e - b_val * b_val

        s = max(0.0, min(1.0, (b_val * f - c * e) / denom)) if abs(denom) > eps else 0.0

        t = (b_val * s + f) / e

        if t < 0.0:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        elif t > 1.0:
            t = 1.0
            s = max(0.0, min(1.0, (b_val - c) / a))

    closest_a = a0 + s * d1
    closest_b = b0 + t * d2
    diff = closest_a - closest_b
    return _dot3(diff, diff)


@njit(cache=True)
def point_triangle_dist_sq(
    p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> float:
    """Compute squared distance from point to triangle."""
    ab = v1 - v0
    ac = v2 - v0
    ap = p - v0

    d1 = _dot3(ab, ap)
    d2 = _dot3(ac, ap)

    if d1 <= 0 and d2 <= 0:
        d = p - v0
        return _dot3(d, d)

    bp = p - v1
    d3 = _dot3(ab, bp)
    d4 = _dot3(ac, bp)

    if d3 >= 0 and d4 <= d3:
        d = p - v1
        return _dot3(d, d)

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        denom = d1 - d3
        v = d1 / denom if denom != 0 else 0.0
        closest = v0 + v * ab
        d = p - closest
        return _dot3(d, d)

    cp = p - v2
    d5 = _dot3(ab, cp)
    d6 = _dot3(ac, cp)

    if d6 >= 0 and d5 <= d6:
        d = p - v2
        return _dot3(d, d)

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        denom = d2 - d6
        w = d2 / denom if denom != 0 else 0.0
        closest = v0 + w * ac
        d = p - closest
        return _dot3(d, d)

    va = d3 * d6 - d5 * d4
    d4_d3 = d4 - d3
    d5_d6 = d5 - d6
    if va <= 0 and d4_d3 >= 0 and d5_d6 >= 0:
        denom = d4_d3 + d5_d6
        w = d4_d3 / denom if denom != 0 else 0.0
        closest = v1 + w * (v2 - v1)
        d = p - closest
        return _dot3(d, d)

    denom = va + vb + vc
    if denom == 0:
        d = p - v0
        return _dot3(d, d)
    v = vb / denom
    w = vc / denom
    closest = v0 + v * ab + w * ac
    d = p - closest
    return _dot3(d, d)


@njit(cache=True)
def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cross product of two 3D vectors."""
    result = np.empty(3, dtype=np.float64)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result


@njit(cache=True)
def edge_triangle_intersect(
    e0: np.ndarray,
    e1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> bool:
    """Test if edge segment intersects triangle using Möller-Trumbore algorithm.

    Returns True if the edge passes through the triangle interior.
    """
    eps = 1e-10
    edge_dir = e1 - e0
    edge_len_sq = _dot3(edge_dir, edge_dir)
    if edge_len_sq < eps * eps:
        return False

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = _cross(edge_dir, edge2)
    a = _dot3(edge1, h)

    if abs(a) < eps:
        return False

    f = 1.0 / a
    s = e0 - v0
    u = f * _dot3(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = _cross(s, edge1)
    v = f * _dot3(edge_dir, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * _dot3(edge2, q)

    # Check if intersection is within edge segment
    # t is already parameterized [0,1] for the segment since edge_dir is unnormalized
    return 0.0 < t < 1.0


@njit(cache=True)
def _segments_intersect_2d(
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    p3x: float,
    p3y: float,
    p4x: float,
    p4y: float,
) -> bool:
    """Check if 2D line segments (p1,p2) and (p3,p4) intersect."""
    d1x = p2x - p1x
    d1y = p2y - p1y
    d2x = p4x - p3x
    d2y = p4y - p3y

    cross = d1x * d2y - d1y * d2x

    if abs(cross) < 1e-14:
        return False

    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / cross
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / cross

    eps = 1e-10
    return eps < t < 1 - eps and eps < u < 1 - eps


@njit(cache=True)
def _point_in_triangle_2d(
    px: float,
    py: float,
    v0x: float,
    v0y: float,
    v1x: float,
    v1y: float,
    v2x: float,
    v2y: float,
) -> bool:
    """Check if point is strictly inside triangle in 2D."""

    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(px, py, v0x, v0y, v1x, v1y)
    d2 = sign(px, py, v1x, v1y, v2x, v2y)
    d3 = sign(px, py, v2x, v2y, v0x, v0y)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


@njit(cache=True)
def triangles_coplanar_overlap(
    t0_v0: np.ndarray,
    t0_v1: np.ndarray,
    t0_v2: np.ndarray,
    t1_v0: np.ndarray,
    t1_v1: np.ndarray,
    t1_v2: np.ndarray,
) -> bool:
    """Check if two coplanar triangles overlap using 2D projection.

    Returns True if triangles overlap (share interior area).
    """
    eps = 1e-10

    # Compute normal of first triangle
    e1 = t0_v1 - t0_v0
    e2 = t0_v2 - t0_v0
    n = _cross(e1, e2)
    n_len = np.sqrt(_dot3(n, n))
    if n_len < eps:
        return False

    # Check if second triangle is coplanar with first
    d0 = abs(_dot3(n, t1_v0 - t0_v0))
    d1 = abs(_dot3(n, t1_v1 - t0_v0))
    d2 = abs(_dot3(n, t1_v2 - t0_v0))

    # Scale threshold by triangle size
    thresh = n_len * eps * 100
    if d0 > thresh or d1 > thresh or d2 > thresh:
        return False  # Not coplanar

    # Project to 2D - choose axis with largest normal component
    an = np.array([abs(n[0]), abs(n[1]), abs(n[2])])
    if an[0] > an[1] and an[0] > an[2]:
        ax, ay = 1, 2
    elif an[1] > an[2]:
        ax, ay = 0, 2
    else:
        ax, ay = 0, 1

    # Triangle 0 vertices in 2D
    u0x, u0y = t0_v0[ax], t0_v0[ay]
    u1x, u1y = t0_v1[ax], t0_v1[ay]
    u2x, u2y = t0_v2[ax], t0_v2[ay]

    # Triangle 1 vertices in 2D
    v0x, v0y = t1_v0[ax], t1_v0[ay]
    v1x, v1y = t1_v1[ax], t1_v1[ay]
    v2x, v2y = t1_v2[ax], t1_v2[ay]

    # Test all edge-edge intersections
    t0_edges = ((u0x, u0y, u1x, u1y), (u1x, u1y, u2x, u2y), (u2x, u2y, u0x, u0y))
    t1_edges = ((v0x, v0y, v1x, v1y), (v1x, v1y, v2x, v2y), (v2x, v2y, v0x, v0y))

    for e0 in t0_edges:
        for e1 in t1_edges:
            if _segments_intersect_2d(
                e0[0], e0[1], e0[2], e0[3], e1[0], e1[1], e1[2], e1[3]
            ):
                return True

    # Check if any vertex of t1 is inside t0
    if _point_in_triangle_2d(v0x, v0y, u0x, u0y, u1x, u1y, u2x, u2y):
        return True
    if _point_in_triangle_2d(v1x, v1y, u0x, u0y, u1x, u1y, u2x, u2y):
        return True
    if _point_in_triangle_2d(v2x, v2y, u0x, u0y, u1x, u1y, u2x, u2y):
        return True

    # Check if any vertex of t0 is inside t1
    if _point_in_triangle_2d(u0x, u0y, v0x, v0y, v1x, v1y, v2x, v2y):
        return True
    if _point_in_triangle_2d(u1x, u1y, v0x, v0y, v1x, v1y, v2x, v2y):
        return True
    return _point_in_triangle_2d(u2x, u2y, v0x, v0y, v1x, v1y, v2x, v2y)


@njit(cache=True)
def elements_share_vertex(
    elem_i: np.ndarray, size_i: int, elem_j: np.ndarray, size_j: int
) -> bool:
    """Check if two elements share any vertex."""
    for i in range(size_i):
        for j in range(size_j):
            if elem_i[i] == elem_j[j]:
                return True
    return False


# =============================================================================
# Edge extraction utilities
# =============================================================================


def extract_unique_edges(F: np.ndarray) -> np.ndarray:
    """Extract unique edges from triangle faces.

    Args:
        F: Triangle faces (M, 3)

    Returns:
        edges: Unique edges (K, 2) with vertex indices sorted per edge
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
        F: Triangle faces (M, 3)

    Returns:
        edges: Unique edges (K, 2)
        edge_to_tri: For each edge, indices of triangles containing it (K, 2).
                     Second index is -1 if edge belongs to only one triangle.
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
    """Container for BVH data of a single element type."""

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
        """Build BVHs for mesh elements.

        Args:
            verts: Vertices (N, 3)
            tris: Triangle faces (M, 3), optional
            edges: Edge segments (K, 2), optional. If None and tris provided,
                   edges are extracted from triangles.
            edge_to_tri: Mapping from edges to parent triangles (K, 2), optional.
            on_flatten: Callback called after flatten phase completes.
            on_build: Callback called after build phase completes.
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

        # Build BVHs in parallel (tree building is sequential, so parallelizing helps)
        self.tri_bvh: Optional[ElementBVH] = None
        self.edge_bvh: Optional[ElementBVH] = None
        self.point_bvh: Optional[ElementBVH] = None

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            if tri_data is not None:
                futures.append(
                    ("tri", executor.submit(self._build_bvh_from_data, *tri_data))
                )
            if edge_data is not None:
                futures.append(
                    ("edge", executor.submit(self._build_bvh_from_data, *edge_data))
                )
            if point_data is not None:
                futures.append(
                    ("point", executor.submit(self._build_bvh_from_data, *point_data))
                )

            for name, future in futures:
                result = future.result()
                if name == "tri":
                    self.tri_bvh = result
                elif name == "edge":
                    self.edge_bvh = result
                else:
                    self.point_bvh = result

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
        """Build BVH from pre-flattened data."""
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
