// File: eigsys_2.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::eigsolve2x2::sym_eigsolve_2x2;
use na::{Matrix2, Matrix3, Matrix3x2, Matrix6, Vector2, Vector6};
use rand::Rng;

const H: f64 = 1e-4;

struct Svd {
    u: Matrix3x2<f64>,
    lambda: Vector2<f64>,
    vt: Matrix2<f64>,
}

fn run_svd(f: &Matrix3x2<f64>) -> Svd {
    let (lambda, v) = sym_eigsolve_2x2(&(f.transpose() * f));
    let mut u = f * v;
    let lambda = lambda.map(|x| (x.max(0.0)).sqrt());
    for i in 0..u.ncols() {
        let u_normalized = u.column(i).normalize();
        u.column_mut(i).copy_from(&u_normalized);
    }
    Svd {
        u,
        lambda,
        vt: v.transpose(),
    }
}

fn energy_s(a: f64, b: f64) -> f64 {
    a.ln().powi(2) + b.ln().powi(2) + a * b + a.powi(2) * b.powi(2)
}

fn energy_f(f: &Matrix3x2<f64>) -> f64 {
    let svd = run_svd(f);
    energy_s(svd.lambda[0], svd.lambda[1])
}

fn approx_grad_s(a: f64, b: f64) -> Vector2<f64> {
    Vector2::new(
        (energy_s(a + H, b) - energy_s(a - H, b)) / (2.0 * H),
        (energy_s(a, b + H) - energy_s(a, b - H)) / (2.0 * H),
    )
}

fn approx_hess_s(a: f64, b: f64) -> Matrix2<f64> {
    let mut h = Matrix2::zeros();
    let g0 = (approx_grad_s(a + H, b) - approx_grad_s(a - H, b)) / (2.0 * H);
    let g1 = (approx_grad_s(a, b + H) - approx_grad_s(a, b - H)) / (2.0 * H);
    h.column_mut(0).copy_from(&g0);
    h.column_mut(1).copy_from(&g1);
    h
}

fn approx_grad_f(f: &Matrix3x2<f64>, d_f: &Matrix3x2<f64>) -> f64 {
    (energy_f(&(f + H * d_f)) - energy_f(&(f - H * d_f))) / (2.0 * H)
}

fn approx_hess_f(f: &Matrix3x2<f64>, d_f: &[Matrix3x2<f64>; 6]) -> Matrix6<f64> {
    let mut h = Matrix6::zeros();
    for i in 0..6 {
        for j in 0..6 {
            let diff = approx_grad_f(&(f + H * d_f[i]), &d_f[j])
                - approx_grad_f(&(f - H * d_f[i]), &d_f[j]);
            h[(i, j)] = diff / (2.0 * H);
        }
    }
    h
}

fn gen_d_f(i: usize, j: usize) -> Matrix3x2<f64> {
    let mut d_f = Matrix3x2::zeros();
    d_f[(i, j)] = 1.0;
    d_f
}

fn mat2vec(a: &Matrix3x2<f64>) -> Vector6<f64> {
    let mut vec = Vector6::zeros();
    let mut idx = 0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            vec[idx] = a[(i, j)];
            idx += 1;
        }
    }
    vec
}

fn expand_u(u: &Matrix3x2<f64>) -> Matrix3<f64> {
    let cross = u.column(0).cross(&u.column(1));
    let mut result = Matrix3::zeros();
    result.column_mut(0).copy_from(&u.column(0));
    result.column_mut(1).copy_from(&u.column(1));
    result.column_mut(2).copy_from(&cross);
    result
}

pub fn run() {
    let mut rng = rand::thread_rng();
    let f = Matrix3x2::from_fn(|_, _| rng.gen_range(-1.0..1.0));
    println!("---- F ----");
    println!("{}", f);

    let d_f: [Matrix3x2<f64>; 6] = [
        gen_d_f(0, 0),
        gen_d_f(1, 0),
        gen_d_f(2, 0),
        gen_d_f(0, 1),
        gen_d_f(1, 1),
        gen_d_f(2, 1),
    ];

    let mut g_f = Matrix3x2::zeros();
    for i in 0..6 {
        g_f[i] = approx_grad_f(&f, &d_f[i]);
    }
    println!("---- numerical gradient ----");
    println!("{:.3e}", g_f);

    let h_f = approx_hess_f(&f, &d_f);
    println!("---- numerical hessian ----");
    println!("{:.3e}", h_f);

    let svd = run_svd(&f);
    let h_s = approx_hess_s(svd.lambda[0], svd.lambda[1]);
    let g_s = approx_grad_s(svd.lambda[0], svd.lambda[1]);
    let (s_s, u_s) = sym_eigsolve_2x2(&h_s);
    let u = expand_u(&svd.u);
    let inv_sqrt = 1.0 / 2.0_f64.sqrt();

    println!("--- analytical gradient ---");
    let mut g_rebuilt = Matrix3x2::zeros();
    for i in 0..6 {
        for k in 0..2 {
            g_rebuilt[i] += g_s[k] * svd.u.column(k).dot(&(d_f[i] * svd.vt.row(k).transpose()));
        }
    }
    println!("{:.3e}", g_rebuilt);

    let q = [
        u * inv_sqrt * Matrix3x2::new(0.0, 1.0, -1.0, 0.0, 0.0, 0.0) * svd.vt,
        u * inv_sqrt * Matrix3x2::new(0.0, 1.0, 1.0, 0.0, 0.0, 0.0) * svd.vt,
        u * Matrix3x2::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0) * svd.vt,
        u * Matrix3x2::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0) * svd.vt,
        u * Matrix3x2::new(u_s[(0, 0)], 0.0, 0.0, u_s[(1, 0)], 0.0, 0.0) * svd.vt,
        u * Matrix3x2::new(u_s[(0, 1)], 0.0, 0.0, u_s[(1, 1)], 0.0, 0.0) * svd.vt,
    ];

    let mut lmds = Vector6::zeros();
    lmds[0] = (g_s[0] + g_s[1]) / (svd.lambda[0] + svd.lambda[1]);
    lmds[1] = if (svd.lambda[0] - svd.lambda[1]).abs() > H {
        (g_s[0] - g_s[1]) / (svd.lambda[0] - svd.lambda[1])
    } else {
        h_s[(0, 0)] - h_s[(0, 1)]
    };
    lmds[2] = g_s[0] / svd.lambda[0];
    lmds[3] = g_s[1] / svd.lambda[1];
    lmds[4] = s_s[0];
    lmds[5] = s_s[1];

    println!("--- analytical hessian ---");
    let mut h_rebuilt = Matrix6::zeros();
    for i in 0..6 {
        if lmds[i] != 0.0 {
            let q_vec = mat2vec(&q[i]);
            h_rebuilt += lmds[i] * q_vec * q_vec.transpose();
        }
    }
    println!("{:.3e}", h_rebuilt);

    println!("--- error ---");
    println!("{:.3e}", (g_f - g_rebuilt).norm() / g_f.norm());
    println!("{:.3e}", (h_f - h_rebuilt).norm() / h_f.norm());
}
