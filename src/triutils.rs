// File: triutils.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0


use nalgebra::{Matrix3xX, Matrix4xX};

pub fn area(x: &Matrix3xX<f32>, faces: &Matrix3xX<usize>, index: usize) -> f32 {
    let f = faces.column(index);
    let x0 = x.column(f[0]);
    let x1 = x.column(f[1]);
    let x2 = x.column(f[2]);
    let e0 = x1 - x0;
    let e1 = x2 - x0;
    0.5 * e0.cross(&e1).norm()
}

pub fn tet_volume(x: &Matrix3xX<f32>, tets: &Matrix4xX<usize>, index: usize) -> f32 {
    let t = tets.column(index);
    let x0 = x.column(t[0]);
    let x1 = x.column(t[1]);
    let x2 = x.column(t[2]);
    let x3 = x.column(t[3]);
    let e0 = x1 - x0;
    let e1 = x2 - x0;
    let e2 = x3 - x0;
    (e0.cross(&e1).dot(&e2)).abs() / 6.0
}

pub fn face_areas(x: &Matrix3xX<f32>, faces: &Matrix3xX<usize>) -> Vec<f32> {
    (0..faces.shape().1).map(|i| area(x, faces, i)).collect()
}

pub fn tet_volumes(x: &Matrix3xX<f32>, tets: &Matrix4xX<usize>) -> Vec<f32> {
    (0..tets.shape().1)
        .map(|i| tet_volume(x, tets, i))
        .collect()
}
