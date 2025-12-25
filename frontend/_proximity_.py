# File: _proximity_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Contact-offset proximity detection using shared BVH acceleration."""

from typing import Optional

import numpy as np

from numba import njit, prange
from tqdm.auto import tqdm

from ._bvh_ import (
    MeshBVH,
    bbox_overlap,
    edge_edge_dist_sq,
    elements_share_vertex,
    point_edge_dist_sq,
    point_triangle_dist_sq,
)


@njit(cache=True)
def _tri_tri_distance_sq(
    verts: np.ndarray, tri_i: np.ndarray, tri_j: np.ndarray
) -> float:
    """Compute minimum distance squared between two triangles."""
    min_dist_sq = np.inf

    t0_i, t1_i, t2_i = verts[tri_i[0]], verts[tri_i[1]], verts[tri_i[2]]
    t0_j, t1_j, t2_j = verts[tri_j[0]], verts[tri_j[1]], verts[tri_j[2]]

    # Point-triangle tests (6 combinations)
    for vi in range(3):
        p = verts[tri_i[vi]]
        dist_sq = point_triangle_dist_sq(p, t0_j, t1_j, t2_j)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    for vj in range(3):
        p = verts[tri_j[vj]]
        dist_sq = point_triangle_dist_sq(p, t0_i, t1_i, t2_i)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    # Edge-edge tests (9 combinations)
    edges = ((0, 1), (1, 2), (2, 0))
    for ei in range(3):
        a0 = verts[tri_i[edges[ei][0]]]
        a1 = verts[tri_i[edges[ei][1]]]
        for ej in range(3):
            b0 = verts[tri_j[edges[ej][0]]]
            b1 = verts[tri_j[edges[ej][1]]]
            dist_sq = edge_edge_dist_sq(a0, a1, b0, b1)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

    return min_dist_sq


@njit(cache=True)
def _tri_edge_distance_sq(
    verts: np.ndarray, tri: np.ndarray, edge: np.ndarray
) -> float:
    """Compute minimum distance squared between triangle and edge."""
    min_dist_sq = np.inf

    t0, t1, t2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
    e0, e1 = verts[edge[0]], verts[edge[1]]

    # Edge endpoints to triangle
    dist_sq = point_triangle_dist_sq(e0, t0, t1, t2)
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
    dist_sq = point_triangle_dist_sq(e1, t0, t1, t2)
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq

    # Triangle vertices to edge
    for vi in range(3):
        p = verts[tri[vi]]
        dist_sq = point_edge_dist_sq(p, e0, e1)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    # Triangle edges to edge
    tri_edges = ((0, 1), (1, 2), (2, 0))
    for ei in range(3):
        a0 = verts[tri[tri_edges[ei][0]]]
        a1 = verts[tri[tri_edges[ei][1]]]
        dist_sq = edge_edge_dist_sq(a0, a1, e0, e1)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    return min_dist_sq


# =============================================================================
# Triangle-Triangle proximity using triangle BVH
# =============================================================================


@njit(cache=True)
def _find_close_tri_tri(
    ti: int,
    verts: np.ndarray,
    tris: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    elem_indices_flat: np.ndarray,
    contact_offset: np.ndarray,
    is_collider: np.ndarray,
    out_pairs: np.ndarray,
    pair_idx: int,
    count_only: bool,
) -> int:
    """Find triangles close to triangle ti using triangle BVH."""
    tri_i = tris[ti]
    offset_i = contact_offset[ti]
    bbox_min_i = tri_bboxes_min[ti] - offset_i
    bbox_max_i = tri_bboxes_max[ti] + offset_i

    count = 0
    stack = np.zeros(64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        if not bbox_overlap(
            bbox_min_i, bbox_max_i, node_bbox_min[node_idx], node_bbox_max[node_idx]
        ):
            continue

        if node_elem_start[node_idx] >= 0:
            start = node_elem_start[node_idx]
            cnt = node_elem_count[node_idx]
            for k in range(cnt):
                tj = elem_indices_flat[start + k]
                if tj <= ti:
                    continue

                if is_collider[ti] and is_collider[tj]:
                    continue

                tri_j = tris[tj]
                if elements_share_vertex(tri_i, 3, tri_j, 3):
                    continue

                offset_j = contact_offset[tj]
                required_dist_sq = (offset_i + offset_j) ** 2

                expanded_min = tri_bboxes_min[tj] - offset_j
                expanded_max = tri_bboxes_max[tj] + offset_j
                if not bbox_overlap(bbox_min_i, bbox_max_i, expanded_min, expanded_max):
                    continue

                dist_sq = _tri_tri_distance_sq(verts, tri_i, tri_j)
                if dist_sq < required_dist_sq:
                    if not count_only:
                        out_pairs[pair_idx + count, 0] = ti
                        out_pairs[pair_idx + count, 1] = tj
                    count += 1
        else:
            stack[stack_ptr] = node_left[node_idx]
            stack_ptr += 1
            stack[stack_ptr] = node_right[node_idx]
            stack_ptr += 1

    return count


# =============================================================================
# Triangle-Edge proximity using edge BVH
# =============================================================================


@njit(cache=True)
def _find_close_tri_edge(
    ti: int,
    verts: np.ndarray,
    tris: np.ndarray,
    edges: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    edge_node_bbox_min: np.ndarray,
    edge_node_bbox_max: np.ndarray,
    edge_node_left: np.ndarray,
    edge_node_right: np.ndarray,
    edge_node_elem_start: np.ndarray,
    edge_node_elem_count: np.ndarray,
    edge_elem_indices_flat: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    tri_offset: np.ndarray,
    edge_offset: np.ndarray,
    tri_is_collider: np.ndarray,
    edge_is_collider: np.ndarray,
    n_tris: int,
    out_pairs: np.ndarray,
    pair_idx: int,
    count_only: bool,
) -> int:
    """Find edges close to triangle ti using edge BVH."""
    tri = tris[ti]
    offset_i = tri_offset[ti]
    bbox_min_i = tri_bboxes_min[ti] - offset_i
    bbox_max_i = tri_bboxes_max[ti] + offset_i

    count = 0
    stack = np.zeros(64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        if not bbox_overlap(
            bbox_min_i, bbox_max_i,
            edge_node_bbox_min[node_idx], edge_node_bbox_max[node_idx]
        ):
            continue

        if edge_node_elem_start[node_idx] >= 0:
            start = edge_node_elem_start[node_idx]
            cnt = edge_node_elem_count[node_idx]
            for k in range(cnt):
                ej = edge_elem_indices_flat[start + k]

                if tri_is_collider[ti] and edge_is_collider[ej]:
                    continue

                edge = edges[ej]
                if elements_share_vertex(tri, 3, edge, 2):
                    continue

                offset_j = edge_offset[ej]
                required_dist_sq = (offset_i + offset_j) ** 2

                expanded_min = edge_bboxes_min[ej] - offset_j
                expanded_max = edge_bboxes_max[ej] + offset_j
                if not bbox_overlap(bbox_min_i, bbox_max_i, expanded_min, expanded_max):
                    continue

                dist_sq = _tri_edge_distance_sq(verts, tri, edge)
                if dist_sq < required_dist_sq:
                    if not count_only:
                        # Triangle index first, then edge index (offset by n_tris)
                        out_pairs[pair_idx + count, 0] = ti
                        out_pairs[pair_idx + count, 1] = n_tris + ej
                    count += 1
        else:
            stack[stack_ptr] = edge_node_left[node_idx]
            stack_ptr += 1
            stack[stack_ptr] = edge_node_right[node_idx]
            stack_ptr += 1

    return count


# =============================================================================
# Edge-Edge proximity using edge BVH
# =============================================================================


@njit(cache=True)
def _find_close_edge_edge(
    ei: int,
    verts: np.ndarray,
    edges: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    elem_indices_flat: np.ndarray,
    contact_offset: np.ndarray,
    is_collider: np.ndarray,
    n_tris: int,
    out_pairs: np.ndarray,
    pair_idx: int,
    count_only: bool,
) -> int:
    """Find edges close to edge ei using edge BVH."""
    edge_i = edges[ei]
    offset_i = contact_offset[ei]
    bbox_min_i = edge_bboxes_min[ei] - offset_i
    bbox_max_i = edge_bboxes_max[ei] + offset_i

    count = 0
    stack = np.zeros(64, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]

        if not bbox_overlap(
            bbox_min_i, bbox_max_i, node_bbox_min[node_idx], node_bbox_max[node_idx]
        ):
            continue

        if node_elem_start[node_idx] >= 0:
            start = node_elem_start[node_idx]
            cnt = node_elem_count[node_idx]
            for k in range(cnt):
                ej = elem_indices_flat[start + k]
                if ej <= ei:
                    continue

                if is_collider[ei] and is_collider[ej]:
                    continue

                edge_j = edges[ej]
                if elements_share_vertex(edge_i, 2, edge_j, 2):
                    continue

                offset_j = contact_offset[ej]
                required_dist_sq = (offset_i + offset_j) ** 2

                expanded_min = edge_bboxes_min[ej] - offset_j
                expanded_max = edge_bboxes_max[ej] + offset_j
                if not bbox_overlap(bbox_min_i, bbox_max_i, expanded_min, expanded_max):
                    continue

                e0_i, e1_i = verts[edge_i[0]], verts[edge_i[1]]
                e0_j, e1_j = verts[edge_j[0]], verts[edge_j[1]]
                dist_sq = edge_edge_dist_sq(e0_i, e1_i, e0_j, e1_j)

                if dist_sq < required_dist_sq:
                    if not count_only:
                        out_pairs[pair_idx + count, 0] = n_tris + ei
                        out_pairs[pair_idx + count, 1] = n_tris + ej
                    count += 1
        else:
            stack[stack_ptr] = node_left[node_idx]
            stack_ptr += 1
            stack[stack_ptr] = node_right[node_idx]
            stack_ptr += 1

    return count


# =============================================================================
# Parallel counting and collection
# =============================================================================


@njit(parallel=True, cache=True)
def _count_tri_tri_parallel(
    verts: np.ndarray,
    tris: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    elem_indices_flat: np.ndarray,
    contact_offset: np.ndarray,
    is_collider: np.ndarray,
    out_counts: np.ndarray,
):
    n_tris = len(tris)
    dummy = np.zeros((1, 2), dtype=np.int32)
    for ti in prange(n_tris):
        out_counts[ti] = _find_close_tri_tri(
            ti, verts, tris, tri_bboxes_min, tri_bboxes_max,
            node_bbox_min, node_bbox_max, node_left, node_right,
            node_elem_start, node_elem_count, elem_indices_flat,
            contact_offset, is_collider, dummy, 0, True,
        )


@njit(parallel=True, cache=True)
def _collect_tri_tri_parallel(
    verts: np.ndarray,
    tris: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    elem_indices_flat: np.ndarray,
    contact_offset: np.ndarray,
    is_collider: np.ndarray,
    pair_offsets: np.ndarray,
    out_pairs: np.ndarray,
):
    n_tris = len(tris)
    for ti in prange(n_tris):
        _find_close_tri_tri(
            ti, verts, tris, tri_bboxes_min, tri_bboxes_max,
            node_bbox_min, node_bbox_max, node_left, node_right,
            node_elem_start, node_elem_count, elem_indices_flat,
            contact_offset, is_collider, out_pairs, pair_offsets[ti], False,
        )


@njit(parallel=True, cache=True)
def _count_tri_edge_parallel(
    verts: np.ndarray,
    tris: np.ndarray,
    edges: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    edge_node_bbox_min: np.ndarray,
    edge_node_bbox_max: np.ndarray,
    edge_node_left: np.ndarray,
    edge_node_right: np.ndarray,
    edge_node_elem_start: np.ndarray,
    edge_node_elem_count: np.ndarray,
    edge_elem_indices_flat: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    tri_offset: np.ndarray,
    edge_offset: np.ndarray,
    tri_is_collider: np.ndarray,
    edge_is_collider: np.ndarray,
    n_tris: int,
    out_counts: np.ndarray,
):
    dummy = np.zeros((1, 2), dtype=np.int32)
    for ti in prange(len(tris)):
        out_counts[ti] = _find_close_tri_edge(
            ti, verts, tris, edges, tri_bboxes_min, tri_bboxes_max,
            edge_node_bbox_min, edge_node_bbox_max, edge_node_left, edge_node_right,
            edge_node_elem_start, edge_node_elem_count, edge_elem_indices_flat,
            edge_bboxes_min, edge_bboxes_max,
            tri_offset, edge_offset, tri_is_collider, edge_is_collider,
            n_tris, dummy, 0, True,
        )


@njit(parallel=True, cache=True)
def _collect_tri_edge_parallel(
    verts: np.ndarray,
    tris: np.ndarray,
    edges: np.ndarray,
    tri_bboxes_min: np.ndarray,
    tri_bboxes_max: np.ndarray,
    edge_node_bbox_min: np.ndarray,
    edge_node_bbox_max: np.ndarray,
    edge_node_left: np.ndarray,
    edge_node_right: np.ndarray,
    edge_node_elem_start: np.ndarray,
    edge_node_elem_count: np.ndarray,
    edge_elem_indices_flat: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    tri_offset: np.ndarray,
    edge_offset: np.ndarray,
    tri_is_collider: np.ndarray,
    edge_is_collider: np.ndarray,
    n_tris: int,
    pair_offsets: np.ndarray,
    out_pairs: np.ndarray,
):
    for ti in prange(len(tris)):
        _find_close_tri_edge(
            ti, verts, tris, edges, tri_bboxes_min, tri_bboxes_max,
            edge_node_bbox_min, edge_node_bbox_max, edge_node_left, edge_node_right,
            edge_node_elem_start, edge_node_elem_count, edge_elem_indices_flat,
            edge_bboxes_min, edge_bboxes_max,
            tri_offset, edge_offset, tri_is_collider, edge_is_collider,
            n_tris, out_pairs, pair_offsets[ti], False,
        )


@njit(parallel=True, cache=True)
def _count_edge_edge_parallel(
    verts: np.ndarray,
    edges: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    elem_indices_flat: np.ndarray,
    contact_offset: np.ndarray,
    is_collider: np.ndarray,
    n_tris: int,
    out_counts: np.ndarray,
):
    n_edges = len(edges)
    dummy = np.zeros((1, 2), dtype=np.int32)
    for ei in prange(n_edges):
        out_counts[ei] = _find_close_edge_edge(
            ei, verts, edges, edge_bboxes_min, edge_bboxes_max,
            node_bbox_min, node_bbox_max, node_left, node_right,
            node_elem_start, node_elem_count, elem_indices_flat,
            contact_offset, is_collider, n_tris, dummy, 0, True,
        )


@njit(parallel=True, cache=True)
def _collect_edge_edge_parallel(
    verts: np.ndarray,
    edges: np.ndarray,
    edge_bboxes_min: np.ndarray,
    edge_bboxes_max: np.ndarray,
    node_bbox_min: np.ndarray,
    node_bbox_max: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    node_elem_start: np.ndarray,
    node_elem_count: np.ndarray,
    elem_indices_flat: np.ndarray,
    contact_offset: np.ndarray,
    is_collider: np.ndarray,
    n_tris: int,
    pair_offsets: np.ndarray,
    out_pairs: np.ndarray,
):
    n_edges = len(edges)
    for ei in prange(n_edges):
        _find_close_edge_edge(
            ei, verts, edges, edge_bboxes_min, edge_bboxes_max,
            node_bbox_min, node_bbox_max, node_left, node_right,
            node_elem_start, node_elem_count, elem_indices_flat,
            contact_offset, is_collider, n_tris,
            out_pairs, pair_offsets[ei], False,
        )


# =============================================================================
# Main API
# =============================================================================


def check_contact_offset_violation(
    V: np.ndarray,
    F: Optional[np.ndarray] = None,
    E: Optional[np.ndarray] = None,
    is_collider: Optional[np.ndarray] = None,
    contact_offset: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> list[tuple[int, int]]:
    """Check for contact offset violations between mesh elements.

    Uses shared MeshBVH with separate triangle and edge BVHs for efficient queries.
    Finds pairs of elements that are closer than the sum of their contact offsets.

    Args:
        V: Vertices (N, 3)
        F: Triangle faces (M, 3), optional
        E: Edge segments (K, 2), optional
        is_collider: Boolean array indicating collider elements.
            Length = M + K (triangles first, then edges).
            Pairs where BOTH elements are colliders are skipped.
        contact_offset: Contact offset per element (M + K,).
            Defaults to 0.0 for all elements.
        verbose: If True, show progress bar.

    Returns:
        List of element pairs (i, j) where distance < offset_i + offset_j.
        Indices: 0..M-1 are triangles, M..M+K-1 are edges.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)

    n_tris = len(F) if F is not None else 0
    n_edges_input = len(E) if E is not None else 0
    n_elems = n_tris + n_edges_input

    if n_elems < 2:
        return []

    # Count steps for progress bar
    steps = ["Flatten", "Building BVH"]
    if n_tris >= 2:
        steps.append("Tri-Tri")
    if n_tris > 0 and n_edges_input > 0:
        steps.append("Tri-Edge")
    if n_edges_input >= 2:
        steps.append("Edge-Edge")

    pbar = tqdm(total=len(steps), desc="separation check", disable=not verbose)

    # Build MeshBVH with callbacks
    pbar.set_postfix_str(steps[0])

    def on_flatten():
        pbar.update(1)
        pbar.set_postfix_str(steps[1])

    def on_build():
        pbar.update(1)

    F_arr = np.ascontiguousarray(F, dtype=np.int32) if F is not None else None
    E_arr = np.ascontiguousarray(E, dtype=np.int32) if E is not None else None
    mesh_bvh = MeshBVH(V, tris=F_arr, edges=E_arr, on_flatten=on_flatten, on_build=on_build)

    # Handle is_collider
    if is_collider is None:
        is_collider_arr = np.zeros(n_elems, dtype=bool)
    else:
        is_collider_arr = np.ascontiguousarray(is_collider, dtype=bool)

    tri_is_collider = is_collider_arr[:n_tris] if n_tris > 0 else np.zeros(0, dtype=bool)
    edge_is_collider = is_collider_arr[n_tris:] if n_edges_input > 0 else np.zeros(0, dtype=bool)

    # Handle contact_offset
    if contact_offset is None:
        offset_arr = np.zeros(n_elems, dtype=np.float64)
    else:
        offset_arr = np.ascontiguousarray(contact_offset, dtype=np.float64)

    tri_offset = offset_arr[:n_tris] if n_tris > 0 else np.zeros(0, dtype=np.float64)
    edge_offset = offset_arr[n_tris:] if n_edges_input > 0 else np.zeros(0, dtype=np.float64)

    all_pairs: list[tuple[int, int]] = []

    # Triangle-Triangle pairs
    if n_tris >= 2 and mesh_bvh.tri_bvh is not None:
        pbar.set_postfix_str("Tri-Tri")
        tri_bvh = mesh_bvh.tri_bvh
        counts = np.zeros(n_tris, dtype=np.int32)
        _count_tri_tri_parallel(
            mesh_bvh.verts, mesh_bvh.tris,
            tri_bvh.elem_bboxes_min, tri_bvh.elem_bboxes_max,
            tri_bvh.node_bbox_min, tri_bvh.node_bbox_max,
            tri_bvh.node_left, tri_bvh.node_right,
            tri_bvh.node_elem_start, tri_bvh.node_elem_count,
            tri_bvh.elem_indices_flat,
            tri_offset, tri_is_collider, counts,
        )
        total = counts.sum()
        if total > 0:
            offsets = np.zeros(n_tris, dtype=np.int32)
            offsets[1:] = np.cumsum(counts[:-1])
            pairs = np.zeros((total, 2), dtype=np.int32)
            _collect_tri_tri_parallel(
                mesh_bvh.verts, mesh_bvh.tris,
                tri_bvh.elem_bboxes_min, tri_bvh.elem_bboxes_max,
                tri_bvh.node_bbox_min, tri_bvh.node_bbox_max,
                tri_bvh.node_left, tri_bvh.node_right,
                tri_bvh.node_elem_start, tri_bvh.node_elem_count,
                tri_bvh.elem_indices_flat,
                tri_offset, tri_is_collider, offsets, pairs,
            )
            all_pairs.extend((int(p[0]), int(p[1])) for p in pairs)
        pbar.update(1)

    # Triangle-Edge pairs
    if n_tris > 0 and n_edges_input > 0 and mesh_bvh.edge_bvh is not None:
        pbar.set_postfix_str("Tri-Edge")
        tri_bvh = mesh_bvh.tri_bvh
        edge_bvh = mesh_bvh.edge_bvh
        assert tri_bvh is not None
        counts = np.zeros(n_tris, dtype=np.int32)
        _count_tri_edge_parallel(
            mesh_bvh.verts, mesh_bvh.tris, mesh_bvh.edges,
            tri_bvh.elem_bboxes_min, tri_bvh.elem_bboxes_max,
            edge_bvh.node_bbox_min, edge_bvh.node_bbox_max,
            edge_bvh.node_left, edge_bvh.node_right,
            edge_bvh.node_elem_start, edge_bvh.node_elem_count,
            edge_bvh.elem_indices_flat,
            edge_bvh.elem_bboxes_min, edge_bvh.elem_bboxes_max,
            tri_offset, edge_offset, tri_is_collider, edge_is_collider,
            n_tris, counts,
        )
        total = counts.sum()
        if total > 0:
            offsets = np.zeros(n_tris, dtype=np.int32)
            offsets[1:] = np.cumsum(counts[:-1])
            pairs = np.zeros((total, 2), dtype=np.int32)
            _collect_tri_edge_parallel(
                mesh_bvh.verts, mesh_bvh.tris, mesh_bvh.edges,
                tri_bvh.elem_bboxes_min, tri_bvh.elem_bboxes_max,
                edge_bvh.node_bbox_min, edge_bvh.node_bbox_max,
                edge_bvh.node_left, edge_bvh.node_right,
                edge_bvh.node_elem_start, edge_bvh.node_elem_count,
                edge_bvh.elem_indices_flat,
                edge_bvh.elem_bboxes_min, edge_bvh.elem_bboxes_max,
                tri_offset, edge_offset, tri_is_collider, edge_is_collider,
                n_tris, offsets, pairs,
            )
            all_pairs.extend((int(p[0]), int(p[1])) for p in pairs)
        pbar.update(1)

    # Edge-Edge pairs
    if n_edges_input >= 2 and mesh_bvh.edge_bvh is not None:
        pbar.set_postfix_str("Edge-Edge")
        edge_bvh = mesh_bvh.edge_bvh
        counts = np.zeros(n_edges_input, dtype=np.int32)
        _count_edge_edge_parallel(
            mesh_bvh.verts, mesh_bvh.edges,
            edge_bvh.elem_bboxes_min, edge_bvh.elem_bboxes_max,
            edge_bvh.node_bbox_min, edge_bvh.node_bbox_max,
            edge_bvh.node_left, edge_bvh.node_right,
            edge_bvh.node_elem_start, edge_bvh.node_elem_count,
            edge_bvh.elem_indices_flat,
            edge_offset, edge_is_collider, n_tris, counts,
        )
        total = counts.sum()
        if total > 0:
            offsets = np.zeros(n_edges_input, dtype=np.int32)
            offsets[1:] = np.cumsum(counts[:-1])
            pairs = np.zeros((total, 2), dtype=np.int32)
            _collect_edge_edge_parallel(
                mesh_bvh.verts, mesh_bvh.edges,
                edge_bvh.elem_bboxes_min, edge_bvh.elem_bboxes_max,
                edge_bvh.node_bbox_min, edge_bvh.node_bbox_max,
                edge_bvh.node_left, edge_bvh.node_right,
                edge_bvh.node_elem_start, edge_bvh.node_elem_count,
                edge_bvh.elem_indices_flat,
                edge_offset, edge_is_collider, n_tris, offsets, pairs,
            )
            all_pairs.extend((int(p[0]), int(p[1])) for p in pairs)
        pbar.update(1)

    pbar.close()
    return all_pairs
