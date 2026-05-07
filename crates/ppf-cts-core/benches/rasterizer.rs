// File: crates/ppf-cts-core/benches/rasterizer.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Criterion bench for the software rasterizer hot path.
// Generates a synthetic scene (a tessellated sphere, scaled to fit
// each target resolution's frame) and renders it through
// `rasterize_triangles` at four resolutions: 256x192, 512x384,
// 1024x768, 2048x1536. The scene is fixed across resolutions so the
// per-pixel cost dominates the per-triangle cost at higher
// resolutions, mirroring the gap reported against Numba.

use criterion::{Criterion, criterion_group, criterion_main};
use ppf_cts_core::kernels::rasterizer::{normals, rasterize_triangles};

fn make_sphere(stacks: usize, slices: usize) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    // Vertex buffer in (N, 3) layout for normals; we'll project them
    // into (N, 4) screen-space once per resolution.
    let mut verts: Vec<f32> = Vec::with_capacity((stacks + 1) * (slices + 1) * 3);
    let mut faces: Vec<i32> = Vec::new();
    let mut colors: Vec<f32> = Vec::new();
    for i in 0..=stacks {
        let phi = std::f32::consts::PI * (i as f32) / (stacks as f32);
        for j in 0..=slices {
            let theta = 2.0 * std::f32::consts::PI * (j as f32) / (slices as f32);
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            verts.push(x);
            verts.push(y);
            verts.push(z);
            // Spread colors over the surface for visual realism.
            colors.push(0.5 + 0.5 * x);
            colors.push(0.5 + 0.5 * y);
            colors.push(0.5 + 0.5 * z);
        }
    }
    let row = (slices + 1) as i32;
    for i in 0..stacks as i32 {
        for j in 0..slices as i32 {
            let a = i * row + j;
            let b = a + row;
            faces.push(a);
            faces.push(b);
            faces.push(a + 1);
            faces.push(b);
            faces.push(b + 1);
            faces.push(a + 1);
        }
    }
    (verts, faces, colors)
}

fn project_to_screen(verts3: &[f32], width: usize, height: usize) -> Vec<f32> {
    // Center, scale to ~80% of frame, drop into the (N, 4) layout
    // expected by the kernel. Z stays in [-1, 1].
    let mut screen = Vec::with_capacity((verts3.len() / 3) * 4);
    let half_w = width as f32 * 0.5;
    let half_h = height as f32 * 0.5;
    let scale = 0.8 * half_w.min(half_h);
    for v in verts3.chunks_exact(3) {
        screen.push(half_w + v[0] * scale);
        screen.push(half_h + v[1] * scale);
        screen.push(v[2]);
        screen.push(1.0);
    }
    screen
}

fn bench_resolution(c: &mut Criterion, label: &str, width: usize, height: usize) {
    // Tessellation tuned so the triangle count is large enough to
    // exercise rayon and large enough that some triangles cover many
    // pixels at 1024x768 and 2048x1536. ~13K triangles.
    let (verts3, faces, colors) = make_sphere(64, 96);
    let normals = normals(&verts3, &faces);
    let screen = project_to_screen(&verts3, width, height);
    let light = [0.3f32, 0.5, 0.8];
    let inv_len = 1.0 / (light[0].powi(2) + light[1].powi(2) + light[2].powi(2)).sqrt();
    let light = [light[0] * inv_len, light[1] * inv_len, light[2] * inv_len];

    let fb_size = width * height * 4;
    let dp_size = width * height;
    let mut fb = vec![255u8; fb_size];
    let mut dp = vec![f32::INFINITY; dp_size];

    c.bench_function(label, |b| {
        b.iter(|| {
            // Reset buffers each iter so depth-test cost is realistic.
            fb.fill(255);
            dp.fill(f32::INFINITY);
            rasterize_triangles(
                &mut fb, &mut dp, width, height,
                &screen, &colors, &normals, &faces,
                light, 0.3,
            );
        });
    });
}

fn benches(c: &mut Criterion) {
    bench_resolution(c, "rasterize_256x192", 256, 192);
    bench_resolution(c, "rasterize_512x384", 512, 384);
    bench_resolution(c, "rasterize_1024x768", 1024, 768);
    bench_resolution(c, "rasterize_2048x1536", 2048, 1536);
}

criterion_group!(rasterizer_benches, benches);
criterion_main!(rasterizer_benches);
