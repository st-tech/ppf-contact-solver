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

    With fixed t=0, 1/3, 2/3, 1 per segment, cp[k] → sampled vertex k*3.
    If both endpoints of a segment are pinned, interior vertices are also pinned.
    """
    pinned_cp = set(cp_indices)
    pinned_sampled = set()

    for s in obj.data.splines:
        is_cyclic = s.use_cyclic_u
        if s.type == "BEZIER":
            n_cp = len(s.bezier_points)
        elif s.type in ("NURBS", "POLY"):
            n_cp = len(s.points)
        else:
            continue

        n_segs = n_cp if is_cyclic else max(1, n_cp - 1)
        for k in range(n_segs):
            k1 = (k + 1) % n_cp
            cp_k_pinned = k in pinned_cp
            cp_k1_pinned = k1 in pinned_cp
            if cp_k_pinned:
                pinned_sampled.add(k * 3)
            if cp_k1_pinned:
                pinned_sampled.add(k1 * 3)
            if cp_k_pinned and cp_k1_pinned:
                pinned_sampled.add(k * 3 + 1)
                pinned_sampled.add(k * 3 + 2)

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

    Each Bezier/NURBS segment is subdivided at t=0, 1/3, 2/3, 1
    (control points + handle projections). No resolution parameter needed.

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

    # Fixed subdivision: t = 0, 1/3, 2/3, 1 per segment (3 edges per segment).
    # Two interior points per segment provides enough data for stable
    # least-squares fitting without regularization.
    seg_t_values = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

    for s in obj.data.splines:
        spline_verts = []
        params = []  # (segment_or_arc_index, local_t) per vertex
        is_cyclic = s.use_cyclic_u

        if s.type == "BEZIER" and len(s.bezier_points) >= 2:
            bps = s.bezier_points
            n_cp = len(bps)
            n_segs = n_cp if is_cyclic else n_cp - 1

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

    With fixed t=0, 1/3, 2/3, 1 per segment, params are fully determined
    by (n_cp, cyclic, type) — no stored metadata needed.
    """
    seg_t_values = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    splines_meta = []
    for s in obj.data.splines:
        is_cyclic = s.use_cyclic_u
        params = []

        if s.type == "BEZIER" and len(s.bezier_points) >= 2:
            n_cp = len(s.bezier_points)
            n_segs = n_cp if is_cyclic else n_cp - 1
            for k in range(n_segs):
                for t in seg_t_values:
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
                for t in seg_t_values:
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


def apply_fit(spline, sim_pos, spline_meta):
    """Fit simulated positions back to control points via least-squares.

    Returns an array of CV positions in Blender's modifier layout:
      Bezier:    [hl_0, co_0, hr_0, hl_1, co_1, hr_1, ...] — 3 per point
      NURBS/Poly: [co_0, co_1, ...] — 1 per point

    The returned array can be written directly to a PC2 file for the
    MESH_CACHE modifier (which accepts curve CVs via AcceptsCVs flag).

    Args:
        spline: Blender spline object
        sim_pos: numpy array (N, 3) of simulated positions in local space
        spline_meta: dict from stored params for this spline

    Returns:
        numpy array of shape (n_cvs, 3) in Blender CV layout order.
    """
    params = spline_meta["params"]
    n_verts = len(params)
    sim_pos = sim_pos[:n_verts]
    stype = spline_meta["type"]

    if stype == "BEZIER":
        n_cp = spline_meta["n_cp"]
        is_cyclic = spline_meta["cyclic"]

        # Fit co positions via least-squares (linear interpolation basis).
        A = np.zeros((n_verts, n_cp))
        for row, (seg, t) in enumerate(params):
            k0 = seg % n_cp
            k1 = (seg + 1) % n_cp
            A[row, k0] += 1.0 - t
            A[row, k1] += t

        co_result = np.zeros((n_cp, 3))
        for ax in range(3):
            co_result[:, ax] = np.linalg.lstsq(A, sim_pos[:, ax], rcond=None)[0]

        # Compute smooth handles (1/6 tangent rule)
        hl = np.zeros((n_cp, 3))
        hr = np.zeros((n_cp, 3))
        for k in range(n_cp):
            co_k = co_result[k]
            if is_cyclic or (k > 0 and k < n_cp - 1):
                prev_co = co_result[(k - 1) % n_cp]
                next_co = co_result[(k + 1) % n_cp]
                tangent = (next_co - prev_co) / 6.0
                hl[k] = co_k - tangent
                hr[k] = co_k + tangent
            elif k == 0:
                tangent = (co_result[1] - co_k) / 3.0
                hr[k] = co_k + tangent
                hl[k] = co_k - tangent
            else:  # k == n_cp - 1
                tangent = (co_k - co_result[n_cp - 2]) / 3.0
                hr[k] = co_k + tangent
                hl[k] = co_k - tangent

        # Pack as [hl_0, co_0, hr_0, hl_1, co_1, hr_1, ...]
        cvs = np.empty((n_cp * 3, 3), dtype=np.float32)
        cvs[0::3] = hl
        cvs[1::3] = co_result
        cvs[2::3] = hr
        return cvs

    elif stype == "NURBS":
        n_cp = spline_meta["n_cp"]
        is_cyclic = spline_meta["cyclic"]
        order = spline_meta["order"]
        degree = order - 1
        step = degree
        wt = spline_meta["weights"]

        A = np.zeros((n_verts, n_cp))
        W = np.zeros(n_verts)
        for row, (arc_idx, t) in enumerate(params):
            arc_indices = [(arc_idx * step + d) % n_cp for d in range(degree + 1)]
            arc_w = [wt[i] for i in arc_indices]
            s_t = 0.0
            for d in range(degree + 1):
                basis = comb(degree, d) * (1 - t) ** (degree - d) * t**d
                bw = basis * arc_w[d]
                A[row, arc_indices[d]] += bw
                s_t += bw
            W[row] = s_t

        result = np.zeros((n_cp, 3), dtype=np.float32)
        for ax in range(3):
            rhs = W * sim_pos[:, ax]
            result[:, ax] = np.linalg.lstsq(A, rhs, rcond=None)[0]

        return result

    elif stype == "POLY":
        n_cp = spline_meta["n_cp"]
        n = min(n_cp, n_verts)
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
