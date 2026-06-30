// File: crates/ppf-cts-core/src/kernels/rasterizer.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Software rasterizer kernels. Three pieces:
//
//   * `rasterize_triangles`: bounding-box scan per triangle, 2D
//     barycentric coords, depth test, vertex-color interpolation,
//     interpolated-normal two-sided diffuse lighting.
//   * `rasterize_lines`: Bresenham + depth-interpolated color, with
//     a square brush of `line_width` pixels.
//   * `normals`: per-vertex normals as the (length-weighted)
//     sum of incident face cross products.
//
// Parallelism: rayon over triangles. Triangles in the same scene that
// touch overlapping pixels race on the depth test, with last-write-wins
// on the contested pixel as the documented semantic.
// The race is implemented via raw pointers wrapped in a
// `Send + Sync` shim; safety relies on the depth+RGBA writes per
// pixel being independent operations that any single thread can
// complete without observing a partially-written neighbor.
//
// Layout convention:
//   * `framebuffer` is `(H, W, 4)` row-major u8, RGBA.
//   * `depth` is `(H, W)` row-major f32. Smaller = nearer.
//   * `screen_verts` is `(N, 4)` row-major f32, `[x, y, z, w]`.
//   * `colors` is `(N, 3)` f32, in `[0, 1]`.
//   * `normals` is `(N, 3)` f32, normalized.
//   * `faces` is `(F, 3)` i32. `segments` is `(S, 2)` i32.

#[inline]
fn fb_idx(width: usize, py: usize, px: usize, c: usize) -> usize {
    (py * width + px) * 4 + c
}

#[inline]
fn dp_idx(width: usize, py: usize, px: usize) -> usize {
    py * width + px
}

/// Raw-pointer shim over the framebuffer + depth slices so the
/// rayon worker threads can write through them. Used only by
/// `rasterize_triangles` / `rasterize_one_triangle`; `rasterize_lines`
/// runs serially through safe slices and does not touch this shim.
///
/// Per-pixel coverage is an intentionally-racy, non-atomic
/// depth-test-then-store: one f32 depth write plus four u8 color
/// writes, i.e. five separate unsynchronized stores guarded by a plain
/// `z < depth` read. When two worker threads cover the same pixel from
/// overlapping triangles, both can read the old depth, both can pass
/// the test, and the stores can interleave so the final depth comes
/// from one triangle while the color bytes come from another. Under
/// Rust's memory model these unsynchronized concurrent reads+writes are
/// a data race, and the documented "last writer wins" is not even
/// self-consistent on a contested pixel. We accept that by design: this
/// is a debug/preview software renderer, the worst case is a handful of
/// mis-colored or mis-depthed pixels at triangle overlaps, and the
/// writes are never out of bounds (px < width / py < height clamps in
/// `rasterize_one_triangle` keep every index valid, see the SAFETY note
/// at the lane-store loop).
struct UnsafeFb {
    fb: *mut u8,
    depth: *mut f32,
    fb_len: usize,
    depth_len: usize,
}

// SAFETY: the per-pixel data race is accepted by design (see UnsafeFb
// docstring). The genuine obligation is that callers keep the
// underlying slices alive for every spawned task: we drive this struct
// with rayon `par_iter` capturing `&UnsafeFb`, so its lifetime is
// bounded by the surrounding function and the slices outlive all tasks.
unsafe impl Send for UnsafeFb {}
unsafe impl Sync for UnsafeFb {}

impl UnsafeFb {
    fn new(fb: &mut [u8], depth: &mut [f32]) -> Self {
        Self {
            fb_len: fb.len(),
            depth_len: depth.len(),
            fb: fb.as_mut_ptr(),
            depth: depth.as_mut_ptr(),
        }
    }

    /// SAFETY: `idx` must be `< self.depth_len`. Caller guarantees
    /// no torn read across pixels.
    #[inline]
    unsafe fn read_depth(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.depth_len);
        *self.depth.add(idx)
    }

    /// SAFETY: same as `read_depth`.
    #[inline]
    unsafe fn write_depth(&self, idx: usize, v: f32) {
        debug_assert!(idx < self.depth_len);
        *self.depth.add(idx) = v;
    }

    /// SAFETY: `off + 3` must be `< self.fb_len`.
    #[inline]
    unsafe fn write_pixel(&self, off: usize, r: u8, g: u8, b: u8, a: u8) {
        debug_assert!(off + 3 < self.fb_len);
        *self.fb.add(off) = r;
        *self.fb.add(off + 1) = g;
        *self.fb.add(off + 2) = b;
        *self.fb.add(off + 3) = a;
    }
}

/// Rasterize triangles with depth testing + diffuse lighting. The
/// framebuffer + depth buffer are mutated in place. Parallelized over
/// triangles via rayon. Overlapping triangles race on the contested
/// pixel (last-write-wins via `UnsafeFb`).
///
/// Hot-loop optimizations vs. the literal port:
///   * Per-row hoist of barycentric setup. `u` and `v` are affine in
///     `px` at fixed `py`, so we precompute the row-start values and
///     the per-column deltas (`du_dx = v1y * inv_denom`,
///     `dv_dx = -v0y * inv_denom`). The inner loop becomes 1 add per
///     coordinate instead of 2 muls + 1 sub.
///   * 8-wide SIMD via the `wide` crate. We compute `u, v, w, z` for
///     8 pixels at a time and form an in-triangle mask, then walk the
///     8 lanes scalarly to do the depth test + write. This keeps the
///     write-side race semantics intact (each pixel is a separate
///     RMW) while letting the heavy float math vectorize. SSE2 / NEON
///     code path is auto-selected by `wide`.
pub fn rasterize_triangles(
    framebuffer: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    screen_verts: &[f32],
    colors: &[f32],
    normals: &[f32],
    faces: &[i32],
    light_dir: [f32; 3],
    ambient: f32,
) {
    use rayon::prelude::*;
    let buffers = UnsafeFb::new(framebuffer, depth);
    let buffers_ref = &buffers;
    let n_faces = faces.len() / 3;

    (0..n_faces).into_par_iter().for_each(|fi| {
        rasterize_one_triangle(
            buffers_ref,
            width,
            height,
            screen_verts,
            colors,
            normals,
            faces,
            fi,
            light_dir,
            ambient,
        );
    });
}

#[inline(always)]
fn rasterize_one_triangle(
    buffers_ref: &UnsafeFb,
    width: usize,
    height: usize,
    screen_verts: &[f32],
    colors: &[f32],
    normals: &[f32],
    faces: &[i32],
    fi: usize,
    light_dir: [f32; 3],
    ambient: f32,
) {
    use wide::{CmpGe, f32x8};

    let i0 = faces[3 * fi] as usize;
    let i1 = faces[3 * fi + 1] as usize;
    let i2 = faces[3 * fi + 2] as usize;

    let x0 = screen_verts[4 * i0];
    let y0 = screen_verts[4 * i0 + 1];
    let z0 = screen_verts[4 * i0 + 2];
    let x1 = screen_verts[4 * i1];
    let y1 = screen_verts[4 * i1 + 1];
    let z1 = screen_verts[4 * i1 + 2];
    let x2 = screen_verts[4 * i2];
    let y2 = screen_verts[4 * i2 + 1];
    let z2 = screen_verts[4 * i2 + 2];

    let min_x = x0.min(x1).min(x2).max(0.0) as i32;
    let max_x = (x0.max(x1).max(x2) as i32 + 1).min((width as i32) - 1);
    let min_y = y0.min(y1).min(y2).max(0.0) as i32;
    let max_y = (y0.max(y1).max(y2) as i32 + 1).min((height as i32) - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    let v0x = x2 - x0;
    let v0y = y2 - y0;
    let v1x = x1 - x0;
    let v1y = y1 - y0;
    let denom = v0x * v1y - v1x * v0y;
    if denom.abs() < 1e-10 {
        return;
    }
    let inv_denom = 1.0 / denom;

    let c0_r = colors[3 * i0];
    let c0_g = colors[3 * i0 + 1];
    let c0_b = colors[3 * i0 + 2];
    let c1_r = colors[3 * i1];
    let c1_g = colors[3 * i1 + 1];
    let c1_b = colors[3 * i1 + 2];
    let c2_r = colors[3 * i2];
    let c2_g = colors[3 * i2 + 1];
    let c2_b = colors[3 * i2 + 2];

    let n0_x = normals[3 * i0];
    let n0_y = normals[3 * i0 + 1];
    let n0_z = normals[3 * i0 + 2];
    let n1_x = normals[3 * i1];
    let n1_y = normals[3 * i1 + 1];
    let n1_z = normals[3 * i1 + 2];
    let n2_x = normals[3 * i2];
    let n2_y = normals[3 * i2 + 1];
    let n2_z = normals[3 * i2 + 2];

    // Per-pixel deltas, hoisted out of the inner loop.
    // u(px, py) = ((px - x0) * v1y - v1x * (py - y0)) * inv_denom
    //          => du/dpx = v1y * inv_denom
    //             du/dpy = -v1x * inv_denom
    // v(px, py) = (v0x * (py - y0) - (px - x0) * v0y) * inv_denom
    //          => dv/dpx = -v0y * inv_denom
    //             dv/dpy = v0x * inv_denom
    let du_dx = v1y * inv_denom;
    let dv_dx = -v0y * inv_denom;
    let du_dy = -v1x * inv_denom;
    let dv_dy = v0x * inv_denom;

    // Lane offsets [0, 1, 2, ..., 7] used to seed each chunk.
    let lane_offsets = f32x8::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let du_dx_v = f32x8::splat(du_dx);
    let dv_dx_v = f32x8::splat(dv_dx);
    let du_step = lane_offsets * du_dx_v;
    let dv_step = lane_offsets * dv_dx_v;
    let chunk_du = f32x8::splat(du_dx * 8.0);
    let chunk_dv = f32x8::splat(dv_dx * 8.0);
    let z0_v = f32x8::splat(z0);
    let z1_v = f32x8::splat(z1);
    let z2_v = f32x8::splat(z2);
    let zero_v = f32x8::splat(0.0);
    let one_v = f32x8::splat(1.0);
    let c0r_v = f32x8::splat(c0_r);
    let c0g_v = f32x8::splat(c0_g);
    let c0b_v = f32x8::splat(c0_b);
    let c1r_v = f32x8::splat(c1_r);
    let c1g_v = f32x8::splat(c1_g);
    let c1b_v = f32x8::splat(c1_b);
    let c2r_v = f32x8::splat(c2_r);
    let c2g_v = f32x8::splat(c2_g);
    let c2b_v = f32x8::splat(c2_b);
    let n0x_v = f32x8::splat(n0_x);
    let n0y_v = f32x8::splat(n0_y);
    let n0z_v = f32x8::splat(n0_z);
    let n1x_v = f32x8::splat(n1_x);
    let n1y_v = f32x8::splat(n1_y);
    let n1z_v = f32x8::splat(n1_z);
    let n2x_v = f32x8::splat(n2_x);
    let n2y_v = f32x8::splat(n2_y);
    let n2z_v = f32x8::splat(n2_z);
    let lx_v = f32x8::splat(light_dir[0]);
    let ly_v = f32x8::splat(light_dir[1]);
    let lz_v = f32x8::splat(light_dir[2]);
    let ambient_v = f32x8::splat(ambient);
    let one_minus_ambient_v = f32x8::splat(1.0 - ambient);
    let v255_v = f32x8::splat(255.0);
    let tiny_v = f32x8::splat(1e-10);

    let dy0 = (min_y as f32) - y0;
    let mut u_row_start = ((min_x as f32) - x0) * du_dx + dy0 * du_dy;
    let mut v_row_start = ((min_x as f32) - x0) * dv_dx + dy0 * dv_dy;

    for py in min_y..=max_y {
        let mut u_chunk = f32x8::splat(u_row_start) + du_step;
        let mut v_chunk = f32x8::splat(v_row_start) + dv_step;
        let mut px = min_x;
        let row_base = (py as usize) * width;
        while px <= max_x {
            let chunk_end_excl = (px + 8).min(max_x + 1);
            let lanes_used = (chunk_end_excl - px) as usize;

            let w_chunk = one_v - u_chunk - v_chunk;
            // Inside-triangle mask. `cmp_ge` returns a lane bitmask
            // (all-ones / all-zeros f32 patterns). Bitwise-and gives
            // the conjunction. Pack to a u8 with one bit per lane via
            // `move_mask` so the dispatch is a single integer test
            // instead of a loop over 8 floats.
            let mask =
                u_chunk.cmp_ge(zero_v) & v_chunk.cmp_ge(zero_v) & w_chunk.cmp_ge(zero_v);
            let mask_bits = mask.move_mask() as u32;
            let valid_mask = if lanes_used == 8 {
                0xFFu32
            } else {
                (1u32 << lanes_used) - 1
            };
            let active = mask_bits & valid_mask;
            if active != 0 {
                // Vectorized depth + color + lighting. We do all the
                // float math 8-wide and only step out to scalar code
                // for the final depth-test/store, which has to be
                // serial to preserve the "last writer wins" race
                // semantics on contested pixels.
                let z_chunk = w_chunk * z0_v + v_chunk * z1_v + u_chunk * z2_v;

                let r_chunk = w_chunk * c0r_v + v_chunk * c1r_v + u_chunk * c2r_v;
                let g_chunk = w_chunk * c0g_v + v_chunk * c1g_v + u_chunk * c2g_v;
                let b_chunk = w_chunk * c0b_v + v_chunk * c1b_v + u_chunk * c2b_v;

                let nx_chunk = w_chunk * n0x_v + v_chunk * n1x_v + u_chunk * n2x_v;
                let ny_chunk = w_chunk * n0y_v + v_chunk * n1y_v + u_chunk * n2y_v;
                let nz_chunk = w_chunk * n0z_v + v_chunk * n1z_v + u_chunk * n2z_v;

                // Two-sided diffuse: intensity = ambient
                //   + (1 - ambient) * |n . L| / max(|n|, tiny).
                // The original code skips the divide when |n| < 1e-10
                // and leaves nx/ny/nz unchanged in that branch. Pick
                // a divisor that is exactly 1.0 when |n| is tiny so
                // we match (degenerate triangles produce zero normals
                // so this branch is effectively dead in practice).
                let n_len_sq = nx_chunk * nx_chunk
                    + ny_chunk * ny_chunk
                    + nz_chunk * nz_chunk;
                let n_len = n_len_sq.sqrt();
                let safe_len = n_len.cmp_ge(tiny_v).blend(n_len, one_v);
                let inv_len = one_v / safe_len;
                let dot = nx_chunk * lx_v + ny_chunk * ly_v + nz_chunk * lz_v;
                let diff = (dot * inv_len).abs();
                let intensity = ambient_v + one_minus_ambient_v * diff;

                let r_lit = (r_chunk * intensity).fast_min(one_v);
                let g_lit = (g_chunk * intensity).fast_min(one_v);
                let b_lit = (b_chunk * intensity).fast_min(one_v);

                let r_byte = r_lit * v255_v;
                let g_byte = g_lit * v255_v;
                let b_byte = b_lit * v255_v;

                let z_arr: [f32; 8] = z_chunk.to_array();
                let r_arr: [f32; 8] = r_byte.to_array();
                let g_arr: [f32; 8] = g_byte.to_array();
                let b_arr: [f32; 8] = b_byte.to_array();

                // Walk only the set lanes via popcount-style trailing
                // zero scan. Avoids a 0..lanes_used loop that always
                // iterates 8 even on sparse coverage.
                let mut bits = active;
                while bits != 0 {
                    let lane = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let z = z_arr[lane];
                    let pxn = (px as usize) + lane;
                    let dpi = row_base + pxn;
                    // SAFETY: pxn is bounded by max_x < width and py
                    // by max_y < height.
                    unsafe {
                        if z < buffers_ref.read_depth(dpi) {
                            buffers_ref.write_depth(dpi, z);
                            let off = dpi * 4;
                            buffers_ref.write_pixel(
                                off,
                                r_arr[lane] as u8,
                                g_arr[lane] as u8,
                                b_arr[lane] as u8,
                                255,
                            );
                        }
                    }
                }
            }
            u_chunk += chunk_du;
            v_chunk += chunk_dv;
            px += 8;
        }
        u_row_start += du_dy;
        v_row_start += dv_dy;
    }
}

/// Bresenham line draw with linear depth/color interpolation and a
/// square brush of `line_width` pixels. Direct port of
/// `_draw_line_bresenham`.
fn draw_line_bresenham(
    framebuffer: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    mut x0: i32,
    mut y0: i32,
    z0: f32,
    x1: i32,
    y1: i32,
    z1: f32,
    r0: f32,
    g0: f32,
    b0: f32,
    r1: f32,
    g1: f32,
    b1: f32,
    line_width: i32,
) {
    let half = line_width / 2;
    let dx_raw = (x1 - x0).abs();
    let dy_raw = (y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let total_steps = dx_raw.max(dy_raw).max(1);

    let mut err = dx_raw - dy_raw;
    let mut step = 0i32;

    loop {
        let t = (step as f32) / (total_steps as f32);
        let z = z0 + t * (z1 - z0);
        let r = r0 + t * (r1 - r0);
        let g = g0 + t * (g1 - g0);
        let b = b0 + t * (b1 - b0);

        for oy in -half..=half {
            for ox in -half..=half {
                let px = x0 + ox;
                let py = y0 + oy;
                if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                    continue;
                }
                let dpi = dp_idx(width, py as usize, px as usize);
                if z < depth[dpi] {
                    depth[dpi] = z;
                    let off = fb_idx(width, py as usize, px as usize, 0);
                    framebuffer[off] = (r.min(1.0) * 255.0) as u8;
                    framebuffer[off + 1] = (g.min(1.0) * 255.0) as u8;
                    framebuffer[off + 2] = (b.min(1.0) * 255.0) as u8;
                    framebuffer[off + 3] = 255;
                }
            }
        }

        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy_raw {
            err -= dy_raw;
            x0 += sx;
        }
        if e2 < dx_raw {
            err += dx_raw;
            y0 += sy;
        }
        step += 1;
    }
}

/// Rasterize line segments with depth-interpolated color. Direct
/// port of `_rasterize_lines`. Unlike `rasterize_triangles`, the line
/// pass is a small overlay/wireframe workload with no parallelism to
/// gain, so it runs serially through safe `&mut [u8]` / `&mut [f32]`
/// slices and deliberately does not use the `UnsafeFb` raw-pointer race
/// machinery. The `z < depth` depth-test semantics match the triangle
/// pass.
pub fn rasterize_lines(
    framebuffer: &mut [u8],
    depth: &mut [f32],
    width: usize,
    height: usize,
    screen_verts: &[f32],
    colors: &[f32],
    segments: &[i32],
    line_width: i32,
) {
    let n_segments = segments.len() / 2;
    for si in 0..n_segments {
        let i0 = segments[2 * si] as usize;
        let i1 = segments[2 * si + 1] as usize;

        let x0 = screen_verts[4 * i0] as i32;
        let y0 = screen_verts[4 * i0 + 1] as i32;
        let z0 = screen_verts[4 * i0 + 2];
        let x1 = screen_verts[4 * i1] as i32;
        let y1 = screen_verts[4 * i1 + 1] as i32;
        let z1 = screen_verts[4 * i1 + 2];

        let r0 = colors[3 * i0];
        let g0 = colors[3 * i0 + 1];
        let b0 = colors[3 * i0 + 2];
        let r1 = colors[3 * i1];
        let g1 = colors[3 * i1 + 1];
        let b1 = colors[3 * i1 + 2];

        draw_line_bresenham(
            framebuffer, depth, width, height, x0, y0, z0, x1, y1, z1, r0, g0, b0, r1, g1, b1,
            line_width,
        );
    }
}

/// Per-vertex normals as the normalized sum of incident face cross
/// products. Face areas implicitly weight the contribution because we
/// don't normalize the per-face normal before accumulating.
pub fn normals(verts_flat: &[f32], faces: &[i32]) -> Vec<f32> {
    let n_verts = verts_flat.len() / 3;
    let mut normals = vec![0.0f32; n_verts * 3];
    let n_faces = faces.len() / 3;
    if n_faces == 0 {
        return normals;
    }
    for fi in 0..n_faces {
        let i0 = faces[3 * fi] as usize;
        let i1 = faces[3 * fi + 1] as usize;
        let i2 = faces[3 * fi + 2] as usize;

        let v0 = [verts_flat[3 * i0], verts_flat[3 * i0 + 1], verts_flat[3 * i0 + 2]];
        let v1 = [verts_flat[3 * i1], verts_flat[3 * i1 + 1], verts_flat[3 * i1 + 2]];
        let v2 = [verts_flat[3 * i2], verts_flat[3 * i2 + 1], verts_flat[3 * i2 + 2]];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        // Cross product, NOT normalized (= face_area * unit_normal).
        let fnx = e1[1] * e2[2] - e1[2] * e2[1];
        let fny = e1[2] * e2[0] - e1[0] * e2[2];
        let fnz = e1[0] * e2[1] - e1[1] * e2[0];

        for &i in &[i0, i1, i2] {
            normals[3 * i] += fnx;
            normals[3 * i + 1] += fny;
            normals[3 * i + 2] += fnz;
        }
    }

    // Per-vertex normalize.
    for i in 0..n_verts {
        let nx = normals[3 * i];
        let ny = normals[3 * i + 1];
        let nz = normals[3 * i + 2];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-10 {
            normals[3 * i] = nx / len;
            normals[3 * i + 1] = ny / len;
            normals[3 * i + 2] = nz / len;
        }
    }
    normals
}

// ---------------------------------------------------------------------------
// `frontend/_render_.py` helpers: pure-compute pieces of the
// `SoftwareRenderer.render` pipeline and the binary-PLY writer used by
// `MitsubaRenderer._export_ply`. Splitting them out lets the Python
// glue stay thin: argparse / Mitsuba scene-dict construction stays in
// Python; matrix math + binary I/O move down here.

// Preview-camera parameters baked into `render_transform`. These match
// the numpy reference render path, so keep them in sync if that path
// changes: the X-axis tilt applied before projection, the extent-fit
// factor that shrinks the normalized mesh inside the view plane, and
// the orthographic near/far depth bounds.
const RENDER_TILT_DEG: f32 = 10.0;
const RENDER_FIT_SCALE: f32 = 0.9;
const RENDER_ORTHO_NEAR: f32 = -10.0;
const RENDER_ORTHO_FAR: f32 = 10.0;

/// 4x4 orthographic projection matrix in row-major order.
pub fn ortho_matrix(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [f32; 16] {
    [
        2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left),
        0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom),
        0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near),
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Run the centering + 10-degree-tilt rotation + extent-normalize +
/// orthographic-projection + screen-space pipeline. Inputs:
///   * `verts_flat`: `(N, 3)` flattened f32 vertices.
///   * `width`, `height`: render target size.
/// Returns `(rotated_verts, screen_verts)` as flat float Vecs:
///   * `rotated_verts` is the post-tilt mesh-space coordinates the
///     normal-computation step needs (Nx3).
///   * `screen_verts` is the raster pipeline's `(N, 4)` `[sx, sy, z, 1]`.
pub fn render_transform(verts_flat: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let n = verts_flat.len() / 3;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // Bounds.
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        for c in 0..3 {
            let v = verts_flat[3 * i + c];
            if v < mn[c] {
                mn[c] = v;
            }
            if v > mx[c] {
                mx[c] = v;
            }
        }
    }
    let bounds = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    let center = [
        bounds[0] / 2.0 + mn[0],
        bounds[1] / 2.0 + mn[1],
        bounds[2] / 2.0 + mn[2],
    ];
    // 10-degree X-axis rotation (matches numpy reference).
    let rad = RENDER_TILT_DEG.to_radians();
    let cos_r = rad.cos();
    let sin_r = rad.sin();
    // Center + rotate.
    let mut rotated = Vec::with_capacity(3 * n);
    for i in 0..n {
        let cx = verts_flat[3 * i] - center[0];
        let cy = verts_flat[3 * i + 1] - center[1];
        let cz = verts_flat[3 * i + 2] - center[2];
        // R @ vec, where R is rotation about X by `rad`:
        //   [1,    0,     0]
        //   [0,  cos, -sin]
        //   [0,  sin,  cos]
        // Numpy code: vert @ R.T → equivalent to R @ vert column-wise.
        let rx = cx;
        let ry = cy * cos_r - cz * sin_r;
        let rz = cy * sin_r + cz * cos_r;
        rotated.push(rx);
        rotated.push(ry);
        rotated.push(rz);
    }
    // Max abs scale across all components.
    let mut max_extent = 0.0f32;
    for &v in &rotated {
        let a = v.abs();
        if a > max_extent {
            max_extent = a;
        }
    }
    if max_extent > 0.0 {
        let s = RENDER_FIT_SCALE / max_extent;
        for v in rotated.iter_mut() {
            *v *= s;
        }
    }
    // Orthographic projection.
    let aspect = (width as f32) / (height as f32);
    let proj = if aspect >= 1.0 {
        ortho_matrix(-aspect, aspect, -1.0, 1.0, RENDER_ORTHO_NEAR, RENDER_ORTHO_FAR)
    } else {
        ortho_matrix(
            -1.0,
            1.0,
            -1.0 / aspect,
            1.0 / aspect,
            RENDER_ORTHO_NEAR,
            RENDER_ORTHO_FAR,
        )
    };
    // Transform to clip space, then NDC, then screen-space.
    let mut screen = Vec::with_capacity(4 * n);
    for i in 0..n {
        let x = rotated[3 * i];
        let y = rotated[3 * i + 1];
        let z = rotated[3 * i + 2];
        // homogeneous w = 1
        let cx = proj[0] * x + proj[1] * y + proj[2] * z + proj[3];
        let cy = proj[4] * x + proj[5] * y + proj[6] * z + proj[7];
        let cz = proj[8] * x + proj[9] * y + proj[10] * z + proj[11];
        let cw = proj[12] * x + proj[13] * y + proj[14] * z + proj[15];
        // perspective divide (orthographic → cw == 1).
        let nx = cx / cw;
        let ny = cy / cw;
        let nz = cz / cw;
        let sx = (nx + 1.0) * 0.5 * (width as f32);
        let sy = (1.0 - ny) * 0.5 * (height as f32);
        screen.push(sx);
        screen.push(sy);
        screen.push(nz);
        screen.push(1.0);
    }
    (rotated, screen)
}

/// Write a binary little-endian PLY to `out`. Per-vertex RGB floats,
/// triangle indices as `list uchar int vertex_indices`.
pub(crate) fn write_ply_binary<W: std::io::Write>(
    mut out: W,
    verts_flat: &[f32],
    colors_flat: &[f32],
    faces: &[i32],
) -> std::io::Result<()> {
    let n_vert = verts_flat.len() / 3;
    let n_face = faces.len() / 3;
    debug_assert!(colors_flat.len() == n_vert * 3);

    // Header.
    out.write_all(b"ply\n")?;
    out.write_all(b"format binary_little_endian 1.0\n")?;
    out.write_all(format!("element vertex {n_vert}\n").as_bytes())?;
    out.write_all(b"property float x\n")?;
    out.write_all(b"property float y\n")?;
    out.write_all(b"property float z\n")?;
    out.write_all(b"property float red\n")?;
    out.write_all(b"property float green\n")?;
    out.write_all(b"property float blue\n")?;
    out.write_all(format!("element face {n_face}\n").as_bytes())?;
    out.write_all(b"property list uchar int vertex_indices\n")?;
    out.write_all(b"end_header\n")?;

    // Vertex data: 3 floats position + 3 floats color, interleaved.
    for i in 0..n_vert {
        for c in 0..3 {
            out.write_all(&verts_flat[3 * i + c].to_le_bytes())?;
        }
        for c in 0..3 {
            out.write_all(&colors_flat[3 * i + c].to_le_bytes())?;
        }
    }
    // Face data: u8 count + 3 i32 indices.
    for fi in 0..n_face {
        out.write_all(&[3u8])?;
        for c in 0..3 {
            out.write_all(&faces[3 * fi + c].to_le_bytes())?;
        }
    }
    Ok(())
}

/// Convenience wrapper that writes the PLY file at `path`.
pub fn write_ply_to_file(
    path: &std::path::Path,
    verts_flat: &[f32],
    colors_flat: &[f32],
    faces: &[i32],
) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut buf = std::io::BufWriter::new(f);
    write_ply_binary(&mut buf, verts_flat, colors_flat, faces)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_buffers(w: usize, h: usize) -> (Vec<u8>, Vec<f32>) {
        let fb = vec![255u8; w * h * 4]; // white background
        let dp = vec![f32::INFINITY; w * h];
        (fb, dp)
    }

    #[test]
    fn normals_empty_returns_zeros() {
        let v = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let f: Vec<i32> = vec![];
        let n = normals(&v, &f);
        assert_eq!(n.len(), v.len());
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn normals_unit_triangle_xy_plane_points_up() {
        // Triangle in XY plane, CCW from above → normal +Z.
        let v: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let f: Vec<i32> = vec![0, 1, 2];
        let n = normals(&v, &f);
        for i in 0..3 {
            assert!((n[3 * i] - 0.0).abs() < 1e-6);
            assert!((n[3 * i + 1] - 0.0).abs() < 1e-6);
            assert!((n[3 * i + 2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rasterize_triangle_writes_into_bounding_box() {
        // 8x8 framebuffer; draw a single red triangle covering the
        // upper-left quarter.
        let w = 8usize;
        let h = 8usize;
        let (mut fb, mut dp) = empty_buffers(w, h);
        // 3 verts in screen space; z=0 for all.
        let sv: Vec<f32> = vec![
            0.0, 0.0, 0.5, 1.0, // v0
            6.0, 0.0, 0.5, 1.0, // v1
            0.0, 6.0, 0.5, 1.0, // v2
        ];
        let colors: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let normals: Vec<f32> = (0..9)
            .map(|i| if i % 3 == 2 { 1.0 } else { 0.0 })
            .collect();
        let faces: Vec<i32> = vec![0, 1, 2];
        rasterize_triangles(
            &mut fb,
            &mut dp,
            w,
            h,
            &sv,
            &colors,
            &normals,
            &faces,
            [0.0, 0.0, 1.0],
            0.5,
        );
        // The (1, 1) pixel should be filled and red-ish (with
        // intensity ~ ambient + (1-ambient)*|nz·1| = 1.0).
        let off = fb_idx(w, 1, 1, 0);
        assert_eq!(fb[off + 3], 255, "pixel (1,1) should have alpha 255");
        assert!(fb[off] > 200, "pixel (1,1) should be red-heavy: r={}", fb[off]);
        // A pixel outside the triangle should still be white.
        let off_outside = fb_idx(w, 7, 7, 0);
        assert_eq!(fb[off_outside], 255);
        assert_eq!(fb[off_outside + 1], 255);
        assert_eq!(fb[off_outside + 2], 255);
    }

    #[test]
    fn rasterize_triangle_passes_depth_test() {
        // Two overlapping triangles; the closer one (smaller z)
        // should win.
        let w = 4usize;
        let h = 4usize;
        let (mut fb, mut dp) = empty_buffers(w, h);
        let sv: Vec<f32> = vec![
            // back triangle (z=0.9), green
            0.0, 0.0, 0.9, 1.0,
            3.0, 0.0, 0.9, 1.0,
            0.0, 3.0, 0.9, 1.0,
            // front triangle (z=0.1), red
            0.0, 0.0, 0.1, 1.0,
            3.0, 0.0, 0.1, 1.0,
            0.0, 3.0, 0.1, 1.0,
        ];
        let colors: Vec<f32> = vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, // green
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // red
        ];
        let normals: Vec<f32> = (0..18).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect();
        // Draw back first, then front.
        let faces1: Vec<i32> = vec![0, 1, 2];
        let faces2: Vec<i32> = vec![3, 4, 5];
        rasterize_triangles(
            &mut fb, &mut dp, w, h, &sv, &colors, &normals, &faces1,
            [0.0, 0.0, 1.0], 0.5,
        );
        rasterize_triangles(
            &mut fb, &mut dp, w, h, &sv, &colors, &normals, &faces2,
            [0.0, 0.0, 1.0], 0.5,
        );
        // Pixel (1, 1) should now be red (front triangle won).
        let off = fb_idx(w, 1, 1, 0);
        assert!(fb[off] > 200 && fb[off + 1] < 50, "expected red, got {}/{}/{}", fb[off], fb[off+1], fb[off+2]);
    }

    #[test]
    fn rasterize_lines_draws_horizontal() {
        let w = 8usize;
        let h = 8usize;
        let (mut fb, mut dp) = empty_buffers(w, h);
        // Two endpoints at (1, 4) → (6, 4), z=0.5, blue.
        let sv: Vec<f32> = vec![
            1.0, 4.0, 0.5, 1.0,
            6.0, 4.0, 0.5, 1.0,
        ];
        let colors: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let segments: Vec<i32> = vec![0, 1];
        rasterize_lines(&mut fb, &mut dp, w, h, &sv, &colors, &segments, 1);
        // Pixels along y=4 from x=1..=6 should be blue.
        for x in 1..=6 {
            let off = fb_idx(w, 4, x, 0);
            assert_eq!(fb[off], 0, "x={x}: r");
            assert_eq!(fb[off + 1], 0);
            assert_eq!(fb[off + 2], 255);
            assert_eq!(fb[off + 3], 255);
        }
        // Outside should still be white.
        let off = fb_idx(w, 0, 0, 0);
        assert_eq!(fb[off], 255);
    }

    #[test]
    fn rasterize_degenerate_triangle_skipped() {
        // Three collinear vertices → denom near zero, must skip.
        let w = 4usize;
        let h = 4usize;
        let (mut fb, mut dp) = empty_buffers(w, h);
        let sv: Vec<f32> = vec![
            0.0, 0.0, 0.5, 1.0,
            2.0, 0.0, 0.5, 1.0,
            4.0, 0.0, 0.5, 1.0,
        ];
        let colors: Vec<f32> = vec![1.0; 9];
        let normals: Vec<f32> = (0..9).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect();
        let faces: Vec<i32> = vec![0, 1, 2];
        rasterize_triangles(
            &mut fb, &mut dp, w, h, &sv, &colors, &normals, &faces,
            [0.0, 0.0, 1.0], 0.5,
        );
        // No pixel should have been touched, still pristine white.
        for i in 0..w * h {
            assert_eq!(fb[4 * i], 255);
            assert_eq!(fb[4 * i + 1], 255);
            assert_eq!(fb[4 * i + 2], 255);
            assert_eq!(fb[4 * i + 3], 255);
        }
    }

    // ---------------------------------------------------------------------
    // _render_.py port tests.

    #[test]
    fn ortho_matrix_diagonals_match_numpy_reference() {
        // From _ortho_matrix(-2, 2, -1, 1, -10, 10).
        let m = ortho_matrix(-2.0, 2.0, -1.0, 1.0, -10.0, 10.0);
        assert!((m[0] - 0.5).abs() < 1e-6);
        assert!((m[5] - 1.0).abs() < 1e-6);
        assert!((m[10] - (-0.1)).abs() < 1e-6);
        assert!((m[15] - 1.0).abs() < 1e-6);
        // Off-axis translations should be zero for symmetric volumes.
        assert_eq!(m[3], 0.0);
        assert_eq!(m[7], 0.0);
        assert_eq!(m[11], 0.0);
    }

    #[test]
    fn render_transform_centers_and_normalizes() {
        // Single-vertex degenerate case: extent is zero, so coords stay
        // at origin and screen lands at the framebuffer center.
        let v = vec![5.0_f32, 7.0, 9.0];
        let (rotated, screen) = render_transform(&v, 100, 100);
        // Centered, no extent → 0,0,0.
        assert!(rotated[0].abs() < 1e-6);
        assert!(rotated[1].abs() < 1e-6);
        assert!(rotated[2].abs() < 1e-6);
        // Screen-space center = (50, 50).
        assert!((screen[0] - 50.0).abs() < 1e-4);
        assert!((screen[1] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn render_transform_screen_y_is_flipped() {
        // Two verts, one above and one below the centroid; after the
        // tilt + projection, the upper vertex should map to a smaller
        // screen-y (top of the framebuffer).
        let v = vec![0.0_f32, 1.0, 0.0,  0.0, -1.0, 0.0];
        let (_rotated, screen) = render_transform(&v, 100, 100);
        // The +y vertex (numpy world up) should land on a smaller
        // screen-y (top of the buffer).
        assert!(screen[1] < screen[5], "upper vert {:.2} not above lower vert {:.2}", screen[1], screen[5]);
    }

    #[test]
    fn render_transform_empty_returns_empty() {
        let (rotated, screen) = render_transform(&[], 100, 100);
        assert!(rotated.is_empty());
        assert!(screen.is_empty());
    }

    #[test]
    fn write_ply_binary_emits_expected_header_and_payload() {
        let verts: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let colors: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let faces: Vec<i32> = vec![0, 1, 2];
        let mut buf: Vec<u8> = Vec::new();
        write_ply_binary(&mut buf, &verts, &colors, &faces).unwrap();
        // Header sanity.
        let header_end = buf.windows(11).position(|w| w == b"end_header\n").unwrap();
        let header = std::str::from_utf8(&buf[..header_end]).unwrap();
        assert!(header.contains("ply\n"));
        assert!(header.contains("format binary_little_endian 1.0\n"));
        assert!(header.contains("element vertex 3\n"));
        assert!(header.contains("element face 1\n"));
        assert!(header.contains("property list uchar int vertex_indices\n"));
        // Payload size: 3 verts * 6 floats * 4 + 1 face * (1 + 3*4)
        let payload_len = buf.len() - header_end - "end_header\n".len();
        assert_eq!(payload_len, 3 * 6 * 4 + (1 + 3 * 4));
    }

    #[test]
    fn write_ply_to_file_roundtrips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("out.ply");
        let verts: Vec<f32> = vec![0.0, 0.0, 0.0];
        let colors: Vec<f32> = vec![0.5, 0.5, 0.5];
        let faces: Vec<i32> = vec![];
        write_ply_to_file(&p, &verts, &colors, &faces).unwrap();
        let body = std::fs::read(&p).unwrap();
        assert!(body.starts_with(b"ply\n"));
    }
}
