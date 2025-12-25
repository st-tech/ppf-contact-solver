# File: _intersection_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Self-intersection detection using edge-triangle intersection tests with BVH."""

from typing import Optional

import numpy as np

from numba import njit, prange
from tqdm.auto import tqdm

from ._bvh_ import (
    MeshBVH,
    bbox_overlap,
    edge_triangle_intersect,
    elements_share_vertex,
    triangles_coplanar_overlap,
)


@njit(cache=True)
def _find_edge_tri_intersections(
    ei: int,
    verts: np.ndarray,
    edges: np.ndarray,
    tris: np.ndarray,
    edge_to_tri: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    tri_node_bbox_min: np.ndarray,
    tri_node_bbox_max: np.ndarray,
    tri_node_left: np.ndarray,
    tri_node_right: np.ndarray,
    tri_node_elem_start: np.ndarray,
    tri_node_elem_count: np.ndarray,
    tri_elem_indices_flat: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    is_collider_tri: np.ndarray,
    out_pairs: np.ndarray,
    pair_idx: int,
    count_only: bool,
) -> int:
    """Find triangles intersected by edge ei using triangle BVH."""
    edge = edges[ei]
    e0 = verts[edge[0]]
    e1 = verts[edge[1]]
    bbox_min_e = edge_bboxes_min[ei]
    bbox_max_e = edge_bboxes_max[ei]

    # Get parent triangles of this edge (to skip)
    parent_tri0 = edge_to_tri[ei, 0]
    parent_tri1 = edge_to_tri[ei, 1]

    # Get parent triangle vertices for coplanar check
    has_parent = parent_tri0 >= 0
    # Initialize to avoid unbound variable (will only be used if has_parent is True)
    parent_tri = tris[0]  # Dummy init for numba
    p0 = e0
    p1 = e1
    p2 = e0
    if has_parent:
        parent_tri = tris[parent_tri0]
        p0 = verts[parent_tri[0]]
        p1 = verts[parent_tri[1]]
        p2 = verts[parent_tri[2]]

    count = 0
    stack = np.zeros(64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        if not bbox_overlap(
            bbox_min_e,
            bbox_max_e,
            tri_node_bbox_min[node_idx],
            tri_node_bbox_max[node_idx],
        ):
            continue

        if tri_node_elem_start[node_idx] >= 0:
            start = tri_node_elem_start[node_idx]
            cnt = tri_node_elem_count[node_idx]
            for k in range(cnt):
                ti = tri_elem_indices_flat[start + k]

                # Skip parent triangles of this edge
                if ti == parent_tri0 or ti == parent_tri1:  # noqa: SIM109
                    continue

                # Skip if both edge's parent triangle and target triangle are colliders
                edge_is_collider = False
                if parent_tri0 >= 0:
                    edge_is_collider = is_collider_tri[parent_tri0]
                if not edge_is_collider and parent_tri1 >= 0:
                    edge_is_collider = is_collider_tri[parent_tri1]

                if edge_is_collider and is_collider_tri[ti]:
                    continue

                # Skip triangles sharing vertices with edge
                tri = tris[ti]
                if elements_share_vertex(edge, 2, tri, 3):
                    continue

                if not bbox_overlap(
                    bbox_min_e, bbox_max_e, tri_bboxes_min[ti], tri_bboxes_max[ti]
                ):
                    continue

                v0 = verts[tri[0]]
                v1 = verts[tri[1]]
                v2 = verts[tri[2]]

                # Check edge-triangle intersection
                intersects = edge_triangle_intersect(e0, e1, v0, v1, v2)

                # If no edge-triangle intersection, check for coplanar overlap
                # But only if parent triangle doesn't share vertices with target
                if (
                    not intersects
                    and has_parent
                    and not elements_share_vertex(parent_tri, 3, tri, 3)
                ):
                    intersects = triangles_coplanar_overlap(p0, p1, p2, v0, v1, v2)

                if intersects:
                    if not count_only:
                        out_pairs[pair_idx + count, 0] = ei
                        out_pairs[pair_idx + count, 1] = ti
                    count += 1
        else:
            stack[stack_ptr] = tri_node_left[node_idx]
            stack_ptr += 1
            stack[stack_ptr] = tri_node_right[node_idx]
            stack_ptr += 1

    return count


@njit(parallel=True, cache=True)
def _count_edge_tri_intersections_parallel(
    verts: np.ndarray,
    edges: np.ndarray,
    tris: np.ndarray,
    edge_to_tri: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    tri_node_bbox_min: np.ndarray,
    tri_node_bbox_max: np.ndarray,
    tri_node_left: np.ndarray,
    tri_node_right: np.ndarray,
    tri_node_elem_start: np.ndarray,
    tri_node_elem_count: np.ndarray,
    tri_elem_indices_flat: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    is_collider_tri: np.ndarray,
    out_counts: np.ndarray,
):
    """Count edge-triangle intersections for each edge in parallel."""
    n_edges = len(edges)
    dummy = np.zeros((1, 2), dtype=np.int32)

    for ei in prange(n_edges):
        out_counts[ei] = _find_edge_tri_intersections(
            ei,
            verts,
            edges,
            tris,
            edge_to_tri,
            edge_bboxes_min,
            edge_bboxes_max,
            tri_node_bbox_min,
            tri_node_bbox_max,
            tri_node_left,
            tri_node_right,
            tri_node_elem_start,
            tri_node_elem_count,
            tri_elem_indices_flat,
            tri_bboxes_min,
            tri_bboxes_max,
            is_collider_tri,
            dummy,
            0,
            True,
        )


@njit(parallel=True, cache=True)
def _collect_edge_tri_intersections_parallel(
    verts: np.ndarray,
    edges: np.ndarray,
    tris: np.ndarray,
    edge_to_tri: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    tri_node_bbox_min: np.ndarray,
    tri_node_bbox_max: np.ndarray,
    tri_node_left: np.ndarray,
    tri_node_right: np.ndarray,
    tri_node_elem_start: np.ndarray,
    tri_node_elem_count: np.ndarray,
    tri_elem_indices_flat: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    is_collider_tri: np.ndarray,
    pair_offsets: np.ndarray,
    out_pairs: np.ndarray,
):
    """Collect edge-triangle intersection pairs in parallel."""
    n_edges = len(edges)

    for ei in prange(n_edges):
        _find_edge_tri_intersections(
            ei,
            verts,
            edges,
            tris,
            edge_to_tri,
            edge_bboxes_min,
            edge_bboxes_max,
            tri_node_bbox_min,
            tri_node_bbox_max,
            tri_node_left,
            tri_node_right,
            tri_node_elem_start,
            tri_node_elem_count,
            tri_elem_indices_flat,
            tri_bboxes_min,
            tri_bboxes_max,
            is_collider_tri,
            out_pairs,
            pair_offsets[ei],
            False,
        )


def _edge_tri_pairs_to_tri_pairs(
    edge_tri_pairs: np.ndarray,
    edge_to_tri: np.ndarray,
) -> list[tuple[int, int]]:
    """Convert edge-triangle intersection pairs to triangle-triangle pairs.

    Each edge belongs to 1-2 triangles. When an edge intersects a triangle,
    we report the pair (edge's parent triangle, intersected triangle).
    """
    tri_pairs_set: set[tuple[int, int]] = set()

    for i in range(len(edge_tri_pairs)):
        ei = edge_tri_pairs[i, 0]
        ti = edge_tri_pairs[i, 1]

        # Get parent triangles of the edge
        parent0 = edge_to_tri[ei, 0]
        parent1 = edge_to_tri[ei, 1]

        if parent0 >= 0 and parent0 != ti:
            pair = (min(parent0, ti), max(parent0, ti))
            tri_pairs_set.add(pair)

        if parent1 >= 0 and parent1 != ti:
            pair = (min(parent1, ti), max(parent1, ti))
            tri_pairs_set.add(pair)

    return sorted(tri_pairs_set)


def check_self_intersection(
    V: np.ndarray,
    F: np.ndarray,
    is_collider: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> list[tuple[int, int]]:
    """Check for self-intersections in a triangle mesh.

    Uses edge-triangle intersection tests: when two triangles intersect,
    an edge from one must pass through the other. BVHs are built for both
    triangles and edges for efficient queries.

    Args:
        V: Vertices (N, 3)
        F: Triangle faces (M, 3)
        is_collider: Optional boolean array (M,) indicating which triangles
            belong to collider meshes. Pairs where BOTH triangles are
            colliders are skipped.
        verbose: If True, show progress bar.

    Returns:
        List of intersecting triangle pairs (i, j) where i < j.
        Empty list if no intersections found.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int32)

    n_tris = len(F)
    if n_tris < 2:
        return []

    steps = ["Flatten", "Building BVH", "Counting intersections", "Collecting pairs"]
    pbar = tqdm(total=len(steps), desc="intersection check", disable=not verbose)

    # Build mesh BVH (triangles + edges)
    pbar.set_postfix_str(steps[0])

    def on_flatten():
        pbar.update(1)
        pbar.set_postfix_str(steps[1])

    def on_build():
        pbar.update(1)

    mesh_bvh = MeshBVH(V, tris=F, on_flatten=on_flatten, on_build=on_build)

    if mesh_bvh.tri_bvh is None or mesh_bvh.edge_bvh is None:
        pbar.close()
        return []

    # Create default is_collider array if not provided
    if is_collider is None:
        is_collider_tri = np.zeros(n_tris, dtype=bool)
    else:
        is_collider_tri = np.ascontiguousarray(is_collider, dtype=bool)

    tri_bvh = mesh_bvh.tri_bvh
    edge_bvh = mesh_bvh.edge_bvh
    edge_to_tri = mesh_bvh.edge_to_tri
    assert edge_to_tri is not None

    n_edges = mesh_bvh.n_edges

    # First pass: count intersections per edge
    pbar.set_postfix_str(steps[2])
    counts = np.zeros(n_edges, dtype=np.int32)
    _count_edge_tri_intersections_parallel(
        mesh_bvh.verts,
        mesh_bvh.edges,
        mesh_bvh.tris,
        edge_to_tri,
        edge_bvh.elem_bboxes_min,
        edge_bvh.elem_bboxes_max,
        tri_bvh.node_bbox_min,
        tri_bvh.node_bbox_max,
        tri_bvh.node_left,
        tri_bvh.node_right,
        tri_bvh.node_elem_start,
        tri_bvh.node_elem_count,
        tri_bvh.elem_indices_flat,
        tri_bvh.elem_bboxes_min,
        tri_bvh.elem_bboxes_max,
        is_collider_tri,
        counts,
    )
    pbar.update(1)

    total_pairs = counts.sum()
    if total_pairs == 0:
        pbar.update(1)
        pbar.close()
        return []

    # Compute offsets
    offsets = np.zeros(n_edges, dtype=np.int32)
    offsets[1:] = np.cumsum(counts[:-1])

    # Second pass: collect pairs
    pbar.set_postfix_str(steps[3])
    edge_tri_pairs = np.zeros((total_pairs, 2), dtype=np.int32)
    _collect_edge_tri_intersections_parallel(
        mesh_bvh.verts,
        mesh_bvh.edges,
        mesh_bvh.tris,
        edge_to_tri,
        edge_bvh.elem_bboxes_min,
        edge_bvh.elem_bboxes_max,
        tri_bvh.node_bbox_min,
        tri_bvh.node_bbox_max,
        tri_bvh.node_left,
        tri_bvh.node_right,
        tri_bvh.node_elem_start,
        tri_bvh.node_elem_count,
        tri_bvh.elem_indices_flat,
        tri_bvh.elem_bboxes_min,
        tri_bvh.elem_bboxes_max,
        is_collider_tri,
        offsets,
        edge_tri_pairs,
    )
    pbar.update(1)
    pbar.close()

    # Convert edge-triangle pairs to triangle-triangle pairs
    return _edge_tri_pairs_to_tri_pairs(edge_tri_pairs, edge_to_tri)
