# File: _bvh_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""AABB-based BVH for triangle mesh nearest point queries using only numpy."""

import numpy as np


class BVHNode:
    """Simple AABB-based BVH node for triangle mesh queries."""

    __slots__ = ["bbox_min", "bbox_max", "left", "right", "tri_indices"]

    def __init__(self):
        self.bbox_min = None
        self.bbox_max = None
        self.left = None
        self.right = None
        self.tri_indices = None  # Leaf node: list of triangle indices


def _build_bvh(
    tri_centroids: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    indices: np.ndarray,
    max_leaf_size: int = 8,
) -> BVHNode:
    """Build a BVH from triangle bounding boxes."""
    node = BVHNode()

    # Compute bounding box for all triangles in this node
    node.bbox_min = tri_bboxes_min[indices].min(axis=0)
    node.bbox_max = tri_bboxes_max[indices].max(axis=0)

    if len(indices) <= max_leaf_size:
        node.tri_indices = indices
        return node

    # Split along the longest axis
    extent = node.bbox_max - node.bbox_min
    axis = np.argmax(extent)

    # Sort by centroid along the split axis
    centroids_axis = tri_centroids[indices, axis]
    sorted_order = np.argsort(centroids_axis)
    sorted_indices = indices[sorted_order]

    mid = len(sorted_indices) // 2
    node.left = _build_bvh(
        tri_centroids, tri_bboxes_min, tri_bboxes_max, sorted_indices[:mid], max_leaf_size
    )
    node.right = _build_bvh(
        tri_centroids, tri_bboxes_min, tri_bboxes_max, sorted_indices[mid:], max_leaf_size
    )

    return node


def _point_to_bbox_dist_sq(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    """Compute squared distance from point to AABB."""
    clamped = np.clip(point, bbox_min, bbox_max)
    diff = point - clamped
    return np.dot(diff, diff)


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Find closest point on triangle ABC to point P. Returns (closest_point, bary_coords)."""
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)

    # Check if P is in vertex region outside A
    if d1 <= 0 and d2 <= 0:
        return a.copy(), np.array([1.0, 0.0, 0.0])

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)

    # Check if P is in vertex region outside B
    if d3 >= 0 and d4 <= d3:
        return b.copy(), np.array([0.0, 1.0, 0.0])

    # Check if P is in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3) if (d1 - d3) != 0 else 0
        return a + v * ab, np.array([1.0 - v, v, 0.0])

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)

    # Check if P is in vertex region outside C
    if d6 >= 0 and d5 <= d6:
        return c.copy(), np.array([0.0, 0.0, 1.0])

    # Check if P is in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6) if (d2 - d6) != 0 else 0
        return a + w * ac, np.array([1.0 - w, 0.0, w])

    # Check if P is in edge region of BC
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6)) if ((d4 - d3) + (d5 - d6)) != 0 else 0
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])

    # P is inside the triangle
    denom = va + vb + vc
    if denom == 0:
        return a.copy(), np.array([1.0, 0.0, 0.0])
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w
    return a + v * ab + w * ac, np.array([u, v, w])


def _query_bvh_closest(
    node: BVHNode,
    point: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    best_dist_sq: float,
    best_tri: int,
    best_bary: np.ndarray,
):
    """Query BVH for closest point to a given point."""
    # Early exit if this node's bbox is farther than current best
    bbox_dist_sq = _point_to_bbox_dist_sq(point, node.bbox_min, node.bbox_max)
    if bbox_dist_sq >= best_dist_sq:
        return best_dist_sq, best_tri, best_bary

    if node.tri_indices is not None:
        # Leaf node: check all triangles
        for ti in node.tri_indices:
            a, b, c = verts[tris[ti, 0]], verts[tris[ti, 1]], verts[tris[ti, 2]]
            closest, bary = _closest_point_on_triangle(point, a, b, c)
            diff = point - closest
            dist_sq = np.dot(diff, diff)
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_tri = ti
                best_bary = bary
    else:
        # Internal node: recurse into children (closer one first)
        left_dist = _point_to_bbox_dist_sq(point, node.left.bbox_min, node.left.bbox_max)
        right_dist = _point_to_bbox_dist_sq(point, node.right.bbox_min, node.right.bbox_max)

        if left_dist < right_dist:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        best_dist_sq, best_tri, best_bary = _query_bvh_closest(
            first, point, verts, tris, best_dist_sq, best_tri, best_bary
        )
        best_dist_sq, best_tri, best_bary = _query_bvh_closest(
            second, point, verts, tris, best_dist_sq, best_tri, best_bary
        )

    return best_dist_sq, best_tri, best_bary


def compute_barycentric_mapping(
    orig_vert: np.ndarray,
    new_vert: np.ndarray,
    new_tri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute barycentric mapping from new surface mesh to original surface vertices.

    For each original surface vertex, find the closest point on the new surface mesh
    and compute its barycentric coordinates within the containing triangle.
    Uses a custom AABB-BVH for O(N log M) performance without external dependencies.

    Args:
        orig_vert: Original surface vertices (N, 3)
        new_vert: New surface vertices from tetrahedralization (P, 3)
        new_tri: New surface triangles from tetrahedralization (Q, 3)

    Returns:
        tri_indices: Triangle index in new mesh for each original vertex (N,)
        bary_coords: Barycentric coordinates for each original vertex (N, 3)
    """
    n_tris = len(new_tri)
    n_orig = len(orig_vert)

    # Compute triangle bounding boxes and centroids
    v0 = new_vert[new_tri[:, 0]]
    v1 = new_vert[new_tri[:, 1]]
    v2 = new_vert[new_tri[:, 2]]

    tri_verts = np.stack([v0, v1, v2], axis=1)  # (n_tris, 3, 3)
    tri_bboxes_min = tri_verts.min(axis=1)
    tri_bboxes_max = tri_verts.max(axis=1)
    tri_centroids = tri_verts.mean(axis=1)

    # Build BVH
    all_indices = np.arange(n_tris, dtype=np.int32)
    bvh_root = _build_bvh(tri_centroids, tri_bboxes_min, tri_bboxes_max, all_indices)

    # Query for each original vertex
    tri_indices = np.zeros(n_orig, dtype=np.int32)
    bary_coords = np.zeros((n_orig, 3), dtype=np.float64)

    for i, p in enumerate(orig_vert):
        _, best_tri, best_bary = _query_bvh_closest(
            bvh_root, p, new_vert, new_tri,
            best_dist_sq=np.inf,
            best_tri=0,
            best_bary=np.array([1.0, 0.0, 0.0]),
        )
        tri_indices[i] = best_tri
        bary_coords[i] = best_bary

    return tri_indices, bary_coords
