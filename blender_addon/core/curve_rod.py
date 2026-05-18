# File: curve_rod.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Curve-to-Rod encode/decode. Single source of truth for
# curve sampling, parameterization, and least-squares fitting.

from math import comb

import numpy as np

from mathutils import Vector  # pyright: ignore


def map_cp_pins_to_sampled(obj, cp_indices):
    """Map control point pin indices to sampled vertex indices.

    ``cp_indices`` carries GLOBAL linear control-point indices across all
    splines, matching the indexing produced by the pin UI op for CURVE
    objects (``ui/dynamics/pin_ops.py``). Sampled vertex indices are
    likewise global across splines, matching ``sample_curve``'s vertex
    concatenation. For Bezier segments where both endpoints are pinned,
    the two interior sample points are pinned too.
    """
    pinned_cp = set(cp_indices)
    pinned_sampled = set()

    cp_offset = 0
    sampled_offset = 0
    for s in obj.data.splines:
        is_cyclic = s.use_cyclic_u
        if s.type == "BEZIER":
            n_cp = len(s.bezier_points)
            # Bezier samples at t = 0, 1 per segment (one sample per CP)
            # so sampled-vertex i corresponds to CP i within the spline.
            n_sampled = n_cp
            stride = 1
        elif s.type in ("NURBS", "POLY"):
            n_cp = len(s.points)
            # POLY: one sample per cp. NURBS: arc-sampled, but the
            # cp-to-sample mapping is not well-defined for higher-order
            # arcs; preserve a 1:1 fallback so single-spline POLY/NURBS
            # callers behave at least as well as before.
            n_sampled = n_cp
            stride = 1
        else:
            continue

        n_segs = n_cp if is_cyclic else max(1, n_cp - 1)
        for k in range(n_segs):
            k1 = (k + 1) % n_cp
            cp_k_pinned = (cp_offset + k) in pinned_cp
            cp_k1_pinned = (cp_offset + k1) in pinned_cp
            if cp_k_pinned:
                pinned_sampled.add(sampled_offset + k * stride)
            if cp_k1_pinned:
                pinned_sampled.add(sampled_offset + k1 * stride)

        cp_offset += n_cp
        sampled_offset += n_sampled

    return sorted(pinned_sampled)


def _eval_bezier(p0, h0, h1, p1, t):
    """Evaluate cubic Bezier at parameter t."""
    t1 = 1.0 - t
    return t1**3 * p0 + 3 * t1**2 * t * h0 + 3 * t1 * t**2 * h1 + t**3 * p1


def _eval_rational_bezier(cps, ws, t):
    """Evaluate rational Bezier of arbitrary degree at parameter t."""
    d = len(cps) - 1
    num = Vector((0, 0, 0))
    den = 0.0
    for i in range(d + 1):
        basis = comb(d, i) * (1 - t) ** (d - i) * t**i
        num += basis * ws[i] * cps[i]
        den += basis * ws[i]
    return num / den if abs(den) > 1e-12 else num



def sample_curve(obj, world_matrix):
    """Sample all splines of a curve object into rod vertices + edges.

    Bezier segments sample at t = 0, 1 (one sample per CP — CPs lie on
    the curve so this is sufficient; edge length matches CP spacing,
    keeping the contact-offset rule contact_offset < edge_length/2 easy
    to satisfy). NURBS arcs sample at t = 0, 1/3, 2/3, 1 because NURBS
    CPs are off-curve and the simulation needs interior points per arc
    to track the curve shape. POLY uses its points directly.

    Args:
        obj: Blender curve object
        world_matrix: 4x4 transformation matrix

    Returns:
        verts: np.array of world-space positions (N, 3) float32
        edges: np.array of edge indices (M, 2) uint32
        params_data: dict to store as metadata (for decode)
    """
    mat = world_matrix
    all_verts = []
    all_edges = []
    splines_meta = []

    # Per-type subdivision. Bezier CPs lie on the curve, so one sample
    # per CP (t = 0, 1) is enough resolution: edge length matches CP
    # spacing, the simulation evolves CP positions directly, and the
    # round-trip fit collapses to identity. NURBS CPs do *not* lie on
    # the curve in general (the curve passes through arcs of the
    # rational Bezier basis), so we still need interior samples per arc
    # for the simulation to track the curve shape; keep t = 0, 1/3,
    # 2/3, 1 there. POLY has one sample per point unconditionally.
    bezier_t_values = [0.0, 1.0]
    nurbs_t_values = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

    for s in obj.data.splines:
        spline_verts = []
        params = []  # (segment_or_arc_index, local_t) per vertex
        is_cyclic = s.use_cyclic_u

        if s.type == "BEZIER" and len(s.bezier_points) >= 2:
            bps = s.bezier_points
            n_cp = len(bps)
            n_segs = n_cp if is_cyclic else n_cp - 1
            seg_t_values = bezier_t_values

            for k in range(n_segs):
                p0 = bps[k].co
                h0 = bps[k].handle_right
                h1 = bps[(k + 1) % n_cp].handle_left
                p1 = bps[(k + 1) % n_cp].co
                for t in seg_t_values:
                    if k > 0 and t == 0.0:
                        continue  # shared with previous segment's t=1
                    if is_cyclic and k == n_segs - 1 and t == 1.0:
                        continue  # shared with first vertex
                    pt = _eval_bezier(p0, h0, h1, p1, t)
                    spline_verts.append(mat @ pt)
                    params.append([k, t])

            splines_meta.append({
                "type": "BEZIER",
                "n_cp": n_cp,
                "cyclic": is_cyclic,
                "params": params,
            })

        elif s.type == "NURBS" and len(s.points) >= 2:
            pts = s.points
            n_cp = len(pts)
            order = s.order_u
            degree = order - 1
            step = degree
            cp = [Vector((p.co[0], p.co[1], p.co[2])) for p in pts]
            wt = [p.co[3] for p in pts]
            n_arcs = n_cp // step if is_cyclic else max(1, (n_cp - 1) // step)
            seg_t_values = nurbs_t_values

            for a in range(n_arcs):
                arc_cp = [cp[(a * step + d) % n_cp] for d in range(degree + 1)]
                arc_w = [wt[(a * step + d) % n_cp] for d in range(degree + 1)]
                for t in seg_t_values:
                    if a > 0 and t == 0.0:
                        continue
                    if is_cyclic and a == n_arcs - 1 and t == 1.0:
                        continue
                    spline_verts.append(mat @ _eval_rational_bezier(arc_cp, arc_w, t))
                    params.append([a, t])

            splines_meta.append({
                "type": "NURBS",
                "n_cp": n_cp,
                "cyclic": is_cyclic,
                "order": order,
                "weights": wt,
                "params": params,
            })

        elif s.type == "POLY" and len(s.points) >= 2:
            for i, p in enumerate(s.points):
                spline_verts.append(mat @ Vector((p.co[0], p.co[1], p.co[2])))
                params.append([i, 0.0])

            splines_meta.append({
                "type": "POLY",
                "n_cp": len(s.points),
                "cyclic": is_cyclic,
                "params": params,
            })

        # Build edges
        if len(spline_verts) >= 2:
            base = len(all_verts)
            for j in range(len(spline_verts) - 1):
                all_edges.append([base + j, base + j + 1])
            if is_cyclic:
                all_edges.append([base + len(spline_verts) - 1, base])
            all_verts.extend(spline_verts)

    verts = np.array(all_verts, dtype=np.float32)
    edges = np.array(all_edges, dtype=np.uint32) if all_edges else np.zeros((0, 2), dtype=np.uint32)
    params_data = {"splines": splines_meta, "n_verts": len(all_verts)}
    return verts, edges, params_data


def compute_params(obj):
    """Compute parameterization from the curve's spline structure.

    Per-type t-values match :func:`sample_curve`: Bezier samples each
    segment at t = 0, 1 (one sample per CP, edge length equals CP
    spacing); NURBS samples each arc at t = 0, 1/3, 2/3, 1 because the
    CPs themselves are off-curve and interior samples are needed for
    the sim to track the arc shape.
    """
    bezier_t_values = [0.0, 1.0]
    nurbs_t_values = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    splines_meta = []
    for s in obj.data.splines:
        is_cyclic = s.use_cyclic_u
        params = []

        if s.type == "BEZIER" and len(s.bezier_points) >= 2:
            n_cp = len(s.bezier_points)
            n_segs = n_cp if is_cyclic else n_cp - 1
            for k in range(n_segs):
                for t in bezier_t_values:
                    if k > 0 and t == 0.0:
                        continue
                    if is_cyclic and k == n_segs - 1 and t == 1.0:
                        continue
                    params.append([k, t])
            splines_meta.append({"type": "BEZIER", "n_cp": n_cp, "cyclic": is_cyclic, "params": params})

        elif s.type == "NURBS" and len(s.points) >= 2:
            n_cp = len(s.points)
            order = s.order_u
            degree = order - 1
            step = degree
            wt = [s.points[i].co[3] for i in range(n_cp)]
            n_arcs = n_cp // step if is_cyclic else max(1, (n_cp - 1) // step)
            for a in range(n_arcs):
                for t in nurbs_t_values:
                    if a > 0 and t == 0.0:
                        continue
                    if is_cyclic and a == n_arcs - 1 and t == 1.0:
                        continue
                    params.append([a, t])
            splines_meta.append({"type": "NURBS", "n_cp": n_cp, "cyclic": is_cyclic, "order": order, "weights": wt, "params": params})

        elif s.type == "POLY" and len(s.points) >= 2:
            for i in range(len(s.points)):
                params.append([i, 0.0])
            splines_meta.append({"type": "POLY", "n_cp": len(s.points), "cyclic": is_cyclic, "params": params})

    return {"splines": splines_meta}


def build_fit_cache(obj):
    """Precompute per-spline data for the per-frame fit.

    Bezier with t = {0, 1} per segment means sampled vertex i corresponds
    directly to control point i within the spline, so the fit is a
    pass-through (no matrix, no SVD). For NURBS we still solve a
    least-squares fit per arc, so we precompute ``pinv(A) * W`` once and
    reuse it across frames; for identical-shape NURBS splines the entry
    is shared (dedup key includes weights). POLY is also a pass-through.

    Returns a list of per-spline cache entries (one per spline of ``obj``,
    aligned with ``compute_params(obj)["splines"]``) plus the matching
    params dict so callers don't need to call ``compute_params`` again.
    """
    params_data = compute_params(obj)

    nurbs_pinv_cache: dict = {}

    cache = []
    for spline_meta in params_data["splines"]:
        stype = spline_meta["type"]
        params = spline_meta["params"]
        n_verts = len(params)
        if stype == "BEZIER":
            cache.append({
                "type": "BEZIER",
                "n_cp": spline_meta["n_cp"],
                "cyclic": bool(spline_meta["cyclic"]),
                "n_verts": n_verts,
            })
        elif stype == "NURBS":
            n_cp = spline_meta["n_cp"]
            order = spline_meta["order"]
            is_cyclic = bool(spline_meta["cyclic"])
            wt = spline_meta["weights"]
            # NURBS A depends on the per-spline weight list; include it
            # in the dedup key so identical-weight splines share.
            key = (n_cp, order, is_cyclic, tuple(wt))
            pinv_AW = nurbs_pinv_cache.get(key)
            if pinv_AW is None:
                degree = order - 1
                step = degree
                A = np.zeros((n_verts, n_cp))
                W = np.zeros(n_verts)
                for row, (arc_idx, t) in enumerate(params):
                    arc_indices = [
                        (arc_idx * step + d) % n_cp for d in range(degree + 1)
                    ]
                    arc_w = [wt[i] for i in arc_indices]
                    s_t = 0.0
                    for d in range(degree + 1):
                        basis = comb(degree, d) * (1 - t) ** (degree - d) * t**d
                        bw = basis * arc_w[d]
                        A[row, arc_indices[d]] += bw
                        s_t += bw
                    W[row] = s_t
                # Pre-fold the per-row weight into the pseudo-inverse so
                # the per-frame call collapses to a single matmul.
                pinv_AW = (np.linalg.pinv(A) * W[None, :]).astype(
                    np.float64, copy=False,
                )
                nurbs_pinv_cache[key] = pinv_AW
            cache.append({
                "type": "NURBS",
                "n_cp": n_cp,
                "n_verts": n_verts,
                "pinv_AW": pinv_AW,
            })
        elif stype == "POLY":
            cache.append({
                "type": "POLY",
                "n_cp": spline_meta["n_cp"],
                "n_verts": n_verts,
            })
        else:
            cache.append(None)
    return cache, params_data


def _bezier_handles_vectorized(co_result, is_cyclic):
    """Vectorized 1/6-tangent handle construction.

    For interior CPs the handle direction is one-sixth of the tangent
    spanning the two neighbours; endpoints of an open spline borrow a
    one-third tangent of the adjacent neighbour. Cyclic splines treat
    every CP as interior via ``np.roll``.
    """
    n_cp = co_result.shape[0]
    if n_cp == 0:
        z = np.empty((0, 3), dtype=co_result.dtype)
        return z, z
    if is_cyclic:
        prev_co = np.roll(co_result, 1, axis=0)
        next_co = np.roll(co_result, -1, axis=0)
        tangent = (next_co - prev_co) / 6.0
        return co_result - tangent, co_result + tangent
    # Open spline: interior uses 1/6, endpoints use 1/3.
    hl = np.empty_like(co_result)
    hr = np.empty_like(co_result)
    if n_cp >= 3:
        prev_co = co_result[:-2]
        next_co = co_result[2:]
        tangent_interior = (next_co - prev_co) / 6.0
        hl[1:-1] = co_result[1:-1] - tangent_interior
        hr[1:-1] = co_result[1:-1] + tangent_interior
    if n_cp >= 2:
        t0 = (co_result[1] - co_result[0]) / 3.0
        hr[0] = co_result[0] + t0
        hl[0] = co_result[0] - t0
        tn = (co_result[-1] - co_result[-2]) / 3.0
        hr[-1] = co_result[-1] + tn
        hl[-1] = co_result[-1] - tn
    else:
        hl[0] = co_result[0]
        hr[0] = co_result[0]
    return hl, hr


def apply_fit_cached(sim_pos, cache_entry):
    """Per-frame fit using a precomputed entry from :func:`build_fit_cache`.

    Returns a ``(n_cvs, 3)`` array in Blender's modifier-cache layout:
    ``[hl_0, co_0, hr_0, hl_1, co_1, hr_1, ...]`` (3 per CP) for Bezier
    and ``[co_0, co_1, ...]`` (1 per CP) for NURBS / POLY.
    """
    if cache_entry is None:
        return np.empty((0, 3), dtype=np.float32)
    stype = cache_entry["type"]
    n_verts = cache_entry["n_verts"]
    sim_pos = sim_pos[:n_verts]

    if stype == "BEZIER":
        # Sampled vertex i == CP i, so the fit is the identity: the
        # simulated positions are the CP positions. Handles are then
        # rebuilt from the 1/6-tangent rule.
        co_result = np.asarray(sim_pos, dtype=np.float64)
        hl, hr = _bezier_handles_vectorized(co_result, cache_entry["cyclic"])
        n_cp = cache_entry["n_cp"]
        cvs = np.empty((n_cp * 3, 3), dtype=np.float32)
        cvs[0::3] = hl
        cvs[1::3] = co_result
        cvs[2::3] = hr
        return cvs
    if stype == "NURBS":
        return (cache_entry["pinv_AW"] @ sim_pos).astype(np.float32, copy=False)
    if stype == "POLY":
        n = min(cache_entry["n_cp"], n_verts)
        return np.array(sim_pos[:n], dtype=np.float32)
    return np.empty((0, 3), dtype=np.float32)


def get_curve_cv_count(obj):
    """Return the total number of CVs for a curve object in modifier layout.

    Bezier: 3 per control point (handle_left, co, handle_right).
    NURBS/Poly: 1 per control point.
    """
    total = 0
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            total += 3 * len(spline.bezier_points)
        else:
            total += len(spline.points)
    return total


def get_curve_rest_cvs(obj):
    """Return the rest-pose CV positions in Blender modifier layout order."""
    cvs = []
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                cvs.append(list(bp.handle_left))
                cvs.append(list(bp.co))
                cvs.append(list(bp.handle_right))
        else:
            for pt in spline.points:
                cvs.append([pt.co[0], pt.co[1], pt.co[2]])
    return np.array(cvs, dtype=np.float32)
