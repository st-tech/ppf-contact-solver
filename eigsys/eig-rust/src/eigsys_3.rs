// File: eigsys_3.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::eigsolve3x3::sym_eigsolve_3x3;
use na::{Matrix, Matrix3, Vector, Vector3};
use rand::Rng;
use std::collections::HashMap;
type Matrix9<T> = Matrix<T, na::U9, na::U9, na::ArrayStorage<T, 9, 9>>;
type Vector9<T> = Vector<T, na::U9, na::ArrayStorage<T, 9, 1>>;

static H: f64 = 1e-4;

fn sqr(x: f64) -> f64 {
    x * x
}

struct Svd {
    u: Matrix3<f64>,
    lambda: Vector3<f64>,
    vt: Matrix3<f64>,
}

fn run_svd(f: &Matrix3<f64>) -> Svd {
    let ft_f = f.transpose() * f;
    let (mut lambda, v) = sym_eigsolve_3x3(&ft_f);
    lambda.iter_mut().for_each(|x| *x = (*x).max(0.0).sqrt());
    let mut u = f * v;
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

fn energy_s(a: f64, b: f64, c: f64, m: &str) -> f64 {
    match m {
        "ARAP" => sqr(a - 1.0) + sqr(b - 1.0) + sqr(c - 1.0),
        "SymDirichlet" => sqr(a) + 1.0 / sqr(a) + sqr(b) + 1.0 / sqr(b) + sqr(c) + 1.0 / sqr(c),
        "MIPS" => (sqr(a) + sqr(b) + sqr(c)) / (a * b * c),
        "Ogden" => (0..5)
            .map(|k| a.powf(0.5f64.powi(k)) + b.powf(0.5f64.powi(k)) + c.powf(0.5f64.powi(k)) - 3.0)
            .sum(),
        "Yeoh" => (0..3)
            .map(|k| (sqr(a) + sqr(b) + sqr(c) - 3.0).powi(k + 1))
            .sum(),
        _ => 0.0,
    }
}

fn energy_f(f: &Matrix3<f64>, m: &str) -> f64 {
    let svd = run_svd(f);
    energy_s(svd.lambda[0], svd.lambda[1], svd.lambda[2], m)
}

fn approx_grad_s(a: f64, b: f64, c: f64, m: &str) -> Vector3<f64> {
    Vector3::new(
        (energy_s(a + H, b, c, m) - energy_s(a - H, b, c, m)) / (2.0 * H),
        (energy_s(a, b + H, c, m) - energy_s(a, b - H, c, m)) / (2.0 * H),
        (energy_s(a, b, c + H, m) - energy_s(a, b, c - H, m)) / (2.0 * H),
    )
}

fn approx_hess_s(a: f64, b: f64, c: f64, m: &str) -> Matrix3<f64> {
    let mut h = Matrix3::zeros();
    let g1 = (approx_grad_s(a + H, b, c, m) - approx_grad_s(a - H, b, c, m)) / (2.0 * H);
    let g2 = (approx_grad_s(a, b + H, c, m) - approx_grad_s(a, b - H, c, m)) / (2.0 * H);
    let g3 = (approx_grad_s(a, b, c + H, m) - approx_grad_s(a, b, c - H, m)) / (2.0 * H);
    h.column_mut(0).copy_from(&g1);
    h.column_mut(1).copy_from(&g2);
    h.column_mut(2).copy_from(&g3);
    h
}

fn approx_grad_f(f: &Matrix3<f64>, df: &Matrix3<f64>, m: &str) -> f64 {
    (energy_f(&(f + H * df), m) - energy_f(&(f - H * df), m)) / (2.0 * H)
}

fn approx_hess_f(f: &Matrix3<f64>, df: &[Matrix3<f64>; 9], m: &str) -> Matrix9<f64> {
    let len = df.len();
    let mut h = Matrix9::<f64>::zeros();
    for i in 0..len {
        for j in 0..len {
            h[(i, j)] = (approx_grad_f(&(f + H * df[i]), &df[j], m)
                - approx_grad_f(&(f - H * df[i]), &df[j], m))
                / (2.0 * H);
        }
    }
    h
}

fn gen_df(i: usize, j: usize) -> Matrix3<f64> {
    let mut df = Matrix3::<f64>::zeros();
    df[(i, j)] = 1.0;
    df
}

pub fn run() {
    let mut rng = rand::thread_rng();
    let f = Matrix3::<f64>::from_iterator((0..9).map(|_| rng.gen_range(-1.0..1.0)));
    let verbose = false;
    let models = ["ARAP", "SymDirichlet", "MIPS", "Ogden", "Yeoh"];

    let mut d_f = [Matrix3::<f64>::zeros(); 9];
    for j in 0..3 {
        for i in 0..3 {
            d_f[i + 3 * j] = gen_df(i, j);
        }
    }

    let svd = run_svd(&f);
    let sigma = svd.lambda;
    let u = svd.u;
    let vt = svd.vt;

    let mut errors = HashMap::new();
    for m in &models {
        let mut g_f = Matrix3::zeros();
        for i in 0..9 {
            g_f[i] = approx_grad_f(&f, &d_f[i], m);
        }
        if verbose {
            println!("---- ({}) numerical gradient ----", m);
            println!("{:.3e}", g_f);
        }
        let h_f = approx_hess_f(&f, &d_f, m);
        if verbose {
            println!("---- ({}) numerical hessian ----", m);
            println!("{:.3e}", h_f);
        }
        let h_s = approx_hess_s(sigma[0], sigma[1], sigma[2], m);
        let g_s = approx_grad_s(sigma[0], sigma[1], sigma[2], m);
        let (s_s, u_s) = sym_eigsolve_3x3(&h_s);

        let mut g_rebuilt = Matrix3::zeros();
        for i in 0..9 {
            for k in 0..3 {
                g_rebuilt[i] += g_s[k] * svd.u.column(k).dot(&(d_f[i] * svd.vt.row(k).transpose()));
            }
        }
        if verbose {
            println!("--- ({}) analytical gradient ---", m);
            println!("{:.3e}", g_rebuilt);
        }

        let inv_sqrt2 = 1.0 / (2.0f64).sqrt();
        let q = [
            inv_sqrt2
                * u
                * Matrix3::<f64>::from_row_slice(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                * vt,
            inv_sqrt2
                * u
                * Matrix3::<f64>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0])
                * vt,
            inv_sqrt2
                * u
                * Matrix3::<f64>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
                * vt,
            inv_sqrt2
                * u
                * Matrix3::<f64>::from_row_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                * vt,
            inv_sqrt2
                * u
                * Matrix3::<f64>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                * vt,
            inv_sqrt2
                * u
                * Matrix3::<f64>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                * vt,
            u * Matrix3::from_diagonal(&u_s.column(0)) * vt,
            u * Matrix3::from_diagonal(&u_s.column(1)) * vt,
            u * Matrix3::from_diagonal(&u_s.column(2)) * vt,
        ];

        let lmds = Vector9::from_iterator([
            (g_s[0] + g_s[1]) / (sigma[0] + sigma[1]),
            (g_s[0] + g_s[2]) / (sigma[0] + sigma[2]),
            (g_s[1] + g_s[2]) / (sigma[1] + sigma[2]),
            if (sigma[0] - sigma[1]).abs() > H {
                (g_s[0] - g_s[1]) / (sigma[0] - sigma[1])
            } else {
                h_s[(0, 0)] - h_s[(0, 1)]
            },
            if (sigma[0] - sigma[2]).abs() > H {
                (g_s[0] - g_s[2]) / (sigma[0] - sigma[2])
            } else {
                h_s[(0, 0)] - h_s[(0, 2)]
            },
            if (sigma[1] - sigma[2]).abs() > H {
                (g_s[1] - g_s[2]) / (sigma[1] - sigma[2])
            } else {
                h_s[(1, 1)] - h_s[(1, 2)]
            },
            s_s[0],
            s_s[1],
            s_s[2],
        ]);

        let mut h_rebuilt = Matrix9::<f64>::zeros();
        for (i, &lmd) in lmds.iter().enumerate() {
            if lmd != 0.0 {
                let q_vec = Vector9::from_iterator(q[i].iter().copied());
                h_rebuilt += lmd * q_vec * q_vec.transpose();
            }
        }

        if verbose {
            println!("--- ({}) analytical hessian ---", m);
            println!("{:.3e}", h_rebuilt);
        }

        errors.insert(
            m.to_string(),
            (
                (g_f - g_rebuilt).norm() / g_f.norm(),
                (h_f - h_rebuilt).norm() / h_f.norm(),
            ),
        );
    }

    println!("===== error summary =====");
    for (name, err) in errors {
        println!("{}: {:.3e} {:.3e}", name, err.0, err.1);
    }
}
