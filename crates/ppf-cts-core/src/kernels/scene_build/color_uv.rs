// File: crates/ppf-cts-core/src/kernels/scene_build/color_uv.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Per-vertex color and per-face UV kernels: dynamic area-strain blend,
// direction/cylinder coloring, pin coverage, direction-based UVs, and
// the shared `hsv_to_rgb` helper.

// ---------------------------------------------------------------------------
// FixedScene.color (dyn-color path): area-strain → HSV blend, with
// face-to-vertex averaging.

/// Compute the per-vertex blended color for the dyn-color hot path.
///
/// Inputs:
///   * `vert`: `(n_verts, 3)` flat row-major (current frame positions).
///   * `tris`: `(n_tris, 3)` flat row-major.
///   * `init_area`: per-triangle initial area (length n_tris).
///   * `face_to_vert_weights`: per-vertex inverse face count, length
///     n_verts.
///   * `dyn_face_color`: per-triangle EnumColor (0 = NONE, !=0 = AREA).
///   * `dyn_face_intensity`: per-triangle blend intensity, length n_tris.
///   * `base_color`: `(n_verts, 3)` per-vertex base color.
///   * `max_area`: ratio cutoff for the blue→red gradient.
///
/// Output is `(n_verts, 3)` flat row-major.
#[allow(clippy::too_many_arguments)]
pub fn dynamic_color(
    vert: &[f64],
    tris: &[[u32; 3]],
    init_area: &[f64],
    face_to_vert_weights: &[f64],
    dyn_face_color: &[u8],
    dyn_face_intensity: &[f64],
    base_color: &[f64],
    max_area: f64,
) -> Vec<f64> {
    let n_verts = face_to_vert_weights.len();
    let n_tris = tris.len();
    assert_eq!(init_area.len(), n_tris);
    assert_eq!(dyn_face_color.len(), n_tris);
    assert_eq!(dyn_face_intensity.len(), n_tris);
    assert_eq!(base_color.len(), 3 * n_verts);
    assert_eq!(vert.len(), 3 * n_verts);

    // Per-face: compute area ratio + face color.
    let mut face_color = vec![0.0f64; 3 * n_tris];
    let mut intensity = vec![0.0f64; n_tris];

    for (ti, tri) in tris.iter().enumerate() {
        let a = tri[0] as usize;
        let b = tri[1] as usize;
        let c = tri[2] as usize;
        let ax = vert[3 * a];
        let ay = vert[3 * a + 1];
        let az = vert[3 * a + 2];
        let bx = vert[3 * b];
        let by = vert[3 * b + 1];
        let bz = vert[3 * b + 2];
        let cx = vert[3 * c];
        let cy = vert[3 * c + 1];
        let cz = vert[3 * c + 2];
        let e1x = bx - ax;
        let e1y = by - ay;
        let e1z = bz - az;
        let e2x = cx - ax;
        let e2y = cy - ay;
        let e2z = cz - az;
        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;
        let area_now = 0.5 * (nx * nx + ny * ny + nz * nz).sqrt();
        let init = init_area[ti];
        let rat = if init > 0.0 { area_now / init } else { 0.0 };
        if dyn_face_color[ti] != 0 {
            let denom = max_area - 1.0;
            let val = if denom > 0.0 {
                ((rat - 1.0) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            intensity[ti] = dyn_face_intensity[ti];
            // colorsys.hsv_to_rgb with hue = 240*(1-val)/360, s=0.75,
            // v=1.0; matches the Python branch.
            let hue = 240.0 * (1.0 - val) / 360.0;
            let (r, g, b) = hsv_to_rgb(hue, 0.75, 1.0);
            face_color[3 * ti] = r;
            face_color[3 * ti + 1] = g;
            face_color[3 * ti + 2] = b;
        }
    }

    // Face → vertex accumulation (np.add.at-style scatter).
    let mut vert_intensity = vec![0.0f64; n_verts];
    let mut vert_face_color = vec![0.0f64; 3 * n_verts];
    for (ti, tri) in tris.iter().enumerate() {
        for &vix in tri.iter() {
            let v = vix as usize;
            vert_intensity[v] += intensity[ti];
            vert_face_color[3 * v] += face_color[3 * ti];
            vert_face_color[3 * v + 1] += face_color[3 * ti + 1];
            vert_face_color[3 * v + 2] += face_color[3 * ti + 2];
        }
    }

    // Multiply by per-vertex weights, then blend.
    let mut out = vec![0.0f64; 3 * n_verts];
    for v in 0..n_verts {
        let w = face_to_vert_weights[v];
        let vi = vert_intensity[v] * w;
        let r = vert_face_color[3 * v] * w;
        let g = vert_face_color[3 * v + 1] * w;
        let b = vert_face_color[3 * v + 2] * w;
        let one_minus_vi = 1.0 - vi;
        out[3 * v] = one_minus_vi * base_color[3 * v] + vi * r;
        out[3 * v + 1] = one_minus_vi * base_color[3 * v + 1] + vi * g;
        out[3 * v + 2] = one_minus_vi * base_color[3 * v + 2] + vi * b;
    }
    out
}

// ---------------------------------------------------------------------------
// Object color helpers.

/// `direction_color`: project verts onto a direction, normalize to
/// `[0, 1]`, then map through `colorsys.hsv_to_rgb(240*(1-y)/360, .75, 1)`.
/// Output is `(N, 3)` flat row-major.
pub fn direction_color(verts: &[f64], direction: [f64; 3]) -> Vec<f64> {
    let n = verts.len() / 3;
    if n == 0 {
        return Vec::new();
    }
    let mut vals = vec![0.0f64; n];
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for i in 0..n {
        let v = verts[3 * i] * direction[0]
            + verts[3 * i + 1] * direction[1]
            + verts[3 * i + 2] * direction[2];
        vals[i] = v;
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    let span = max_val - min_val;
    let inv = if span > 0.0 { 1.0 / span } else { 0.0 };
    let mut out = vec![0.0f64; 3 * n];
    for i in 0..n {
        let y = (vals[i] - min_val) * inv;
        let hue = 240.0 * (1.0 - y) / 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.75, 1.0);
        out[3 * i] = r;
        out[3 * i + 1] = g;
        out[3 * i + 2] = b;
    }
    out
}

/// `cylinder_color`: project verts into the (ex, ey) plane around
/// `center`, take `atan2(y, x)`, normalize to `[0, 1)`, and HSV-map.
/// `direction` and `up` define the local frame `ey = up`,
/// `ex = direction cross ey`.
pub fn cylinder_color(
    verts: &[f64],
    center: [f64; 3],
    direction: [f64; 3],
    up: [f64; 3],
) -> Vec<f64> {
    let n = verts.len() / 3;
    if n == 0 {
        return Vec::new();
    }
    let ey = up;
    // ex = direction × ey
    let ex = [
        direction[1] * ey[2] - direction[2] * ey[1],
        direction[2] * ey[0] - direction[0] * ey[2],
        direction[0] * ey[1] - direction[1] * ey[0],
    ];
    let mut out = vec![0.0f64; 3 * n];
    let two_pi = std::f64::consts::TAU;
    for i in 0..n {
        let dx = verts[3 * i] - center[0];
        let dy = verts[3 * i + 1] - center[1];
        let dz = verts[3 * i + 2] - center[2];
        let xv = dx * ex[0] + dy * ex[1] + dz * ex[2];
        let yv = dx * ey[0] + dy * ey[1] + dz * ey[2];
        let mut angle = yv.atan2(xv);
        // Match Python's `np.mod(angle, 2*pi) / (2*pi)`.
        angle = angle.rem_euclid(two_pi) / two_pi;
        let (r, g, b) = hsv_to_rgb(angle, 0.75, 1.0);
        out[3 * i] = r;
        out[3 * i + 1] = g;
        out[3 * i + 2] = b;
    }
    out
}

// ---------------------------------------------------------------------------
// Per-vertex pin coverage check (`Object.update_static`).

/// Returns true iff every vertex in `[0, n_verts)` appears at least
/// once in any of the `pin_indices` lists.
pub fn all_vertices_pinned(n_verts: usize, pin_indices: &[&[i64]]) -> bool {
    if n_verts == 0 {
        return false;
    }
    let mut flag = vec![false; n_verts];
    for inds in pin_indices {
        for &i in inds.iter() {
            if i >= 0 && (i as usize) < n_verts {
                flag[i as usize] = true;
            }
        }
    }
    flag.iter().all(|&f| f)
}

// ---------------------------------------------------------------------------
// Object.direction: build per-face UVs from two orthogonal directions.
//
// Validates that ex/ey are orthogonal and orthogonal to every face
// normal, then emits a per-face `(3, 2)` UV row by projecting the
// triangle vertices onto `ex` and `ey`.

#[derive(Debug, thiserror::Error)]
pub enum DirectionError {
    #[error("ex and ey must be orthogonal. ex: [{ex:?}], ey: [{ey:?}]")]
    NotOrthogonal { ex: [f64; 3], ey: [f64; 3] },
    #[error("ex must be orthogonal to the face normal. normal: [{normal:?}]")]
    ExNotOrthogonalToNormal { normal: [f64; 3] },
    #[error("ey must be orthogonal to the face normal. normal: [{normal:?}]")]
    EyNotOrthogonalToNormal { normal: [f64; 3] },
}

/// Build per-face UVs via `(a.dot(ex), a.dot(ey))` for each triangle
/// vertex. Returns one `[u_a, v_a, u_b, v_b, u_c, v_c]` row per face.
/// `eps` is the epsilon used for the orthogonality checks.
pub fn uv_from_directions(
    verts: &[f64],
    tris: &[[u32; 3]],
    ex_in: [f64; 3],
    ey_in: [f64; 3],
    eps: f64,
) -> Result<Vec<f64>, DirectionError> {
    let nx = (ex_in[0] * ex_in[0] + ex_in[1] * ex_in[1] + ex_in[2] * ex_in[2]).sqrt();
    let ny = (ey_in[0] * ey_in[0] + ey_in[1] * ey_in[1] + ey_in[2] * ey_in[2]).sqrt();
    let inv_x = if nx > 0.0 { 1.0 / nx } else { 0.0 };
    let inv_y = if ny > 0.0 { 1.0 / ny } else { 0.0 };
    let ex = [ex_in[0] * inv_x, ex_in[1] * inv_x, ex_in[2] * inv_x];
    let ey = [ey_in[0] * inv_y, ey_in[1] * inv_y, ey_in[2] * inv_y];
    if (ex[0] * ey[0] + ex[1] * ey[1] + ex[2] * ey[2]).abs() > eps {
        return Err(DirectionError::NotOrthogonal { ex, ey });
    }
    let mut out = Vec::with_capacity(6 * tris.len());
    for tri in tris {
        let ai = tri[0] as usize;
        let bi = tri[1] as usize;
        let ci = tri[2] as usize;
        let a = [verts[3 * ai], verts[3 * ai + 1], verts[3 * ai + 2]];
        let b = [verts[3 * bi], verts[3 * bi + 1], verts[3 * bi + 2]];
        let c = [verts[3 * ci], verts[3 * ci + 1], verts[3 * ci + 2]];
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let nm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let inv_n = if nm > 0.0 { 1.0 / nm } else { 0.0 };
        n[0] *= inv_n;
        n[1] *= inv_n;
        n[2] *= inv_n;
        // Python's check uses the original (un-normalized) `_ex` /
        // `_ey` against `n`, so mirror that exactly.
        if (n[0] * ex_in[0] + n[1] * ex_in[1] + n[2] * ex_in[2]).abs() > eps {
            return Err(DirectionError::ExNotOrthogonalToNormal { normal: n });
        }
        if (n[0] * ey_in[0] + n[1] * ey_in[1] + n[2] * ey_in[2]).abs() > eps {
            return Err(DirectionError::EyNotOrthogonalToNormal { normal: n });
        }
        out.push(a[0] * ex[0] + a[1] * ex[1] + a[2] * ex[2]);
        out.push(a[0] * ey[0] + a[1] * ey[1] + a[2] * ey[2]);
        out.push(b[0] * ex[0] + b[1] * ex[1] + b[2] * ex[2]);
        out.push(b[0] * ey[0] + b[1] * ey[1] + b[2] * ey[2]);
        out.push(c[0] * ex[0] + c[1] * ex[1] + c[2] * ex[2]);
        out.push(c[0] * ey[0] + c[1] * ey[1] + c[2] * ey[2]);
    }
    Ok(out)
}

/// Standard HSV→RGB. `hue` is in `[0, 1]` (callers divide by 360).
fn hsv_to_rgb(hue: f64, sat: f64, val: f64) -> (f64, f64, f64) {
    if sat == 0.0 {
        return (val, val, val);
    }
    let mut h = hue * 6.0;
    if h >= 6.0 {
        h -= 6.0;
    }
    let i = h.floor();
    let f = h - i;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
    let t = val * (1.0 - sat * (1.0 - f));
    match i as i32 % 6 {
        0 => (val, t, p),
        1 => (q, val, p),
        2 => (p, val, t),
        3 => (p, q, val),
        4 => (t, p, val),
        _ => (val, p, q),
    }
}
