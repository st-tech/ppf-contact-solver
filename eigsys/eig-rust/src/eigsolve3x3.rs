// File: eigsolve3x3.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use na::{Matrix3, Vector3};
use std::f64::consts::PI;

#[allow(non_snake_case)]
fn eigvalues(A: &Matrix3<f64>) -> Option<Vector3<f64>> {
    let p1 = A[(0, 1)].powi(2) + A[(0, 2)].powi(2) + A[(1, 2)].powi(2);
    let q = A.trace() / 3.0;
    let p2 = (A[(0, 0)] - q).powi(2) + (A[(1, 1)] - q).powi(2) + (A[(2, 2)] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p.abs() < 1e-8 {
        None
    } else {
        let B = (1.0 / p) * (A - q * Matrix3::identity());
        let r = B.determinant() / 2.0;
        let phi = if r <= -1.0 {
            PI / 3.0
        } else if r >= 1.0 {
            0.0
        } else {
            r.acos() / 3.0
        };
        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + 2.0 * PI / 3.0).cos();
        let eig2 = 3.0 * q - eig1 - eig3;
        Some(Vector3::new(eig1, eig2, eig3))
    }
}

fn pick_largest(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> Vector3<f64> {
    let (a_norm, b_norm, c_norm) = (a.norm_squared(), b.norm_squared(), c.norm_squared());
    if a_norm > b_norm {
        if a_norm > c_norm {
            *a
        } else {
            *c
        }
    } else if b_norm > c_norm {
        *b
    } else {
        *c
    }
}

#[allow(non_snake_case)]
fn find_ortho(A: &Matrix3<f64>) -> (Vector3<f64>, Option<Vector3<f64>>) {
    let eps = 1e-8;
    let (u, v, w) = (A.column(0), A.column(1), A.column(2));
    let (uv, vw, wu) = (u.cross(&v), v.cross(&w), w.cross(&u));
    let q = pick_largest(&uv, &vw, &wu);
    if q.norm_squared() < eps {
        let p = pick_largest(&u.into(), &v.into(), &w.into());
        let x = p.cross(&Vector3::new(1.0, 0.0, 0.0));
        let x = if x.norm_squared() < eps {
            p.cross(&Vector3::new(0.0, 1.0, 0.0))
        } else {
            x
        };
        let y = p.cross(&x);
        (x.normalize(), Some(y.normalize()))
    } else {
        (q.normalize(), None)
    }
}

#[allow(non_snake_case)]
fn eigvectors(A: &Matrix3<f64>, lmd: &Vector3<f64>) -> Matrix3<f64> {
    let uv = find_ortho(&(A - lmd[0] * Matrix3::identity()));
    if let Some(v) = uv.1 {
        Matrix3::from_columns(&[uv.0, v, uv.0.cross(&v)])
    } else {
        let tmp = find_ortho(&(A - lmd[1] * Matrix3::identity()));
        let v = tmp.0;
        Matrix3::from_columns(&[uv.0, v, uv.0.cross(&v)])
    }
}

#[allow(non_snake_case)]
pub fn sym_eigsolve_3x3(A: &Matrix3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
    let scale = A.norm();
    let B = A / scale;
    if let Some(lmd) = eigvalues(&B) {
        (scale * lmd, eigvectors(&B, &lmd))
    } else {
        let val = scale * (B.column(0).norm() + B.column(1).norm() + B.column(2).norm()) / 3.0;
        (Vector3::new(val, val, val), Matrix3::identity())
    }
}
