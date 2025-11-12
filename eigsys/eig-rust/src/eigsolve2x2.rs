// File: eigsolve2x2.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use na::{Matrix2, Vector2};

#[allow(non_snake_case)]
fn eigvalues(A: &Matrix2<f64>) -> Vector2<f64> {
    let a00 = A[(0, 0)];
    let a01 = A[(0, 1)];
    let a11 = A[(1, 1)];
    let d = ((a00 - a11).powi(2) + 4.0 * a01 * a01).sqrt() / 2.0;
    Vector2::new((a00 + a11) / 2.0 - d, (a00 + a11) / 2.0 + d)
}

fn rot90(x: &Vector2<f64>) -> Vector2<f64> {
    Vector2::new(x[1], -x[0])
}

#[allow(non_snake_case)]
fn find_ortho(A: &Matrix2<f64>, x: &Vector2<f64>) -> Vector2<f64> {
    let eps = 1e-8;
    let u = rot90(&A.column(0).into());
    let v = rot90(&A.column(1).into());
    if u.norm_squared() > eps * eps {
        u.normalize()
    } else if v.norm_squared() > eps * eps {
        v.normalize()
    } else {
        rot90(x)
    }
}

#[allow(non_snake_case)]
pub fn eigvectors(A: &Matrix2<f64>, lmd: &Vector2<f64>) -> Matrix2<f64> {
    let u = find_ortho(&(A - Matrix2::identity() * lmd[0]), &Vector2::new(0.0, 1.0));
    let v = find_ortho(&(A - Matrix2::identity() * lmd[1]), &-u);
    Matrix2::from_columns(&[u, v])
}

#[allow(non_snake_case)]
pub fn sym_eigsolve_2x2(A: &Matrix2<f64>) -> (Vector2<f64>, Matrix2<f64>) {
    let scale = A.norm();
    let B = A / scale;
    let lmd = eigvalues(&B);
    let eigvecs = eigvectors(&B, &lmd);
    (scale * lmd, eigvecs)
}
