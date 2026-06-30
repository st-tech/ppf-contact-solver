// File: pd_arap.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Simple, stable, implicit elastic solver for the CUDA-free emulator
// (libsimbackend_cpu). It exists ONLY so cloth deformation (and the
// mesh-tearing pipeline that keys off FEM strain) can be exercised on a
// macOS / no-GPU host. It is NOT a port of the production solver.
//
// Method: Projective Dynamics (Bouaziz et al. 2014), local-global form,
// with the membrane As-Rigid-As-Possible (ARAP) energy ONLY:
//
//     E_i(x) = (w_i / 2) * || F_i(x) - R_i ||_F^2 ,
//
// where F_i is the 3x2 shell deformation gradient
// (utility::compute_deformation_grad: F = [x1-x0, x2-x0] * inv_rest2x2,
// matching cpp/utility/utility.cu) and R_i is the closest 3x2 frame with
// orthonormal columns (the ARAP local step: SVD of F, singular values
// projected to 1). This is the same ARAP energy the production model uses
// (cpp/energy/model/arap.hpp: 0.5*mu*sum (sigma_k - 1)^2), minus the
// detsqr area term and minus bending/contact, kept deliberately minimal.
//
// Integrator: implicit Euler. The global step solves
//
//     (M/dt^2 + sum_i w_i S_i^T S_i) x = M/dt^2 y + sum_i w_i S_i^T R_i ,
//
// with y = x_t + dt v_t + dt^2 g the inertial + gravity predictor. The
// left matrix is constant for a fixed topology, fixed-vertex set, and dt,
// so it is Cholesky-prefactored once and reused across frames; it is
// refactored only when one of those changes (e.g. after a tear grows the
// vertex count). Kinematic pins are Dirichlet boundary conditions,
// condensed into the right-hand side during assembly. Unconditionally
// stable regardless of stiffness, which is why an implicit method is used.
//
// Disabled by default (preserves the historical kinematic-only emulator).
// Opt in with PPF_EMULATED_ELASTIC=1. PPF_EMULATED_ELASTIC_ITERS sets the
// number of local-global iterations per step (default 20).
//
// Scope / limitations (deliberately minimal; this is a test enabler):
//   * Membrane ARAP on shell faces only. No bending, no tets/rods, no
//     contact/collision, no strain limiting, no plasticity, no damping
//     beyond the implicit integrator's numerical dissipation.
//   * The SVD local step extracts the closest orthonormal-column frame but
//     has NO inversion / reflection recovery. Under smooth loading (gravity,
//     gradual pin motion) elements stay non-inverted and the solve relaxes
//     to the correct rest state; an instantaneous boundary "snap" large
//     enough to invert an element can latch into a reflected local minimum.
//     The intended drivers (drape, slow stretch, tearing) load gradually.

#ifndef PD_ARAP_HPP
#define PD_ARAP_HPP

#include "../cpp/data.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cstdlib>
#include <vector>

namespace pd_arap {

using Mat32d = Eigen::Matrix<double, 3, 2>;
using Mat33d = Eigen::Matrix<double, 3, 3>;
using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;

inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *e = std::getenv("PPF_EMULATED_ELASTIC");
        cached = (e && *e && e[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

inline int iterations() {
    static int cached = -1;
    if (cached < 0) {
        const char *e = std::getenv("PPF_EMULATED_ELASTIC_ITERS");
        cached = (e && *e) ? std::atoi(e) : 20;
        if (cached < 1) {
            cached = 1;
        }
    }
    return cached;
}

// Per shell face: the three vertex indices, the ARAP weight w_i, and the
// constant 3x2 operator B mapping one coordinate of the three vertices to
// that coordinate's row of F (F_row = [p0 p1 p2] * B). Rest-shape derived,
// so rebuilt only when topology changes.
struct FacePre {
    int v[3];
    double w;
    Mat32d B;
};

struct Solver {
    int n = 0;                 // vertex count this factorization was built for
    double dt = 0.0;           // dt this factorization was built for
    std::vector<FacePre> faces;
    std::vector<double> mass;  // lumped per-vertex mass
    std::vector<char> is_fixed;
    std::vector<int> reduced;  // vertex -> free-DOF index, or -1 if fixed
    int n_free = 0;
    Eigen::SimplicialLDLT<SpMat> chol;
    bool factored = false;
    std::size_t fixed_sig = 0; // hash of the fixed-vertex set
};

inline Solver &state() {
    static Solver s;
    return s;
}

inline std::size_t hash_fixed(const std::vector<char> &is_fixed) {
    std::size_t h = 1469598103934665603ull; // FNV-1a
    for (char c : is_fixed) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }
    return h;
}

// Rebuild the per-face rest-shape operators. Called when vertex/face count
// changes (initialize, and after a future tear edits topology).
inline void rebuild_faces(const DataSet &d) {
    Solver &s = state();
    s.faces.clear();
    Mat32d Dm;
    Dm << -1.0, -1.0, 1.0, 0.0, 0.0, 1.0; // p -> [p1-p0, p2-p0]
    const unsigned nface = d.shell_face_count;
    s.faces.reserve(nface);
    for (unsigned i = 0; i < nface; ++i) {
        const Vec3u f = d.mesh.mesh.face.data[i];
        const Mat2x2f ir = d.inv_rest2x2.data[i];
        Eigen::Matrix2d ird;
        ird << ir(0, 0), ir(0, 1), ir(1, 0), ir(1, 1);
        FacePre fp;
        fp.v[0] = static_cast<int>(f[0]);
        fp.v[1] = static_cast<int>(f[1]);
        fp.v[2] = static_cast<int>(f[2]);
        fp.B = Dm * ird;
        const double area = d.prop.face.data[i].area;
        const unsigned pidx = d.prop.face.data[i].param_index;
        const double mu = d.param_arrays.face.data[pidx].mu;
        fp.w = (mu > 0.0 ? mu : 1.0) * (area > 0.0 ? area : 1.0);
        s.faces.push_back(fp);
    }
    s.mass.assign(d.prop.vertex.size, 1.0);
    for (unsigned i = 0; i < d.prop.vertex.size; ++i) {
        const double m = d.prop.vertex.data[i].mass;
        s.mass[i] = (m > 0.0) ? m : 1.0;
    }
}

// Assemble M/dt^2 + L over the FREE DOFs only (Dirichlet condensation),
// then Cholesky-prefactor. Reused until topology / fixed-set / dt changes.
inline void factor(double dt) {
    Solver &s = state();
    const int n = s.n;
    s.reduced.assign(n, -1);
    s.n_free = 0;
    for (int v = 0; v < n; ++v) {
        if (!s.is_fixed[v]) {
            s.reduced[v] = s.n_free++;
        }
    }
    const double inv_dt2 = 1.0 / (dt * dt);
    std::vector<Trip> trips;
    trips.reserve(s.faces.size() * 9 + s.n_free);
    for (int v = 0; v < n; ++v) {
        if (!s.is_fixed[v]) {
            trips.emplace_back(s.reduced[v], s.reduced[v], s.mass[v] * inv_dt2);
        }
    }
    for (const FacePre &fp : s.faces) {
        const Mat33d Ke = fp.w * (fp.B * fp.B.transpose());
        for (int a = 0; a < 3; ++a) {
            const int va = fp.v[a];
            if (s.is_fixed[va]) {
                continue;
            }
            for (int b = 0; b < 3; ++b) {
                const int vb = fp.v[b];
                if (s.is_fixed[vb]) {
                    continue; // fixed columns are condensed into the RHS
                }
                trips.emplace_back(s.reduced[va], s.reduced[vb], Ke(a, b));
            }
        }
    }
    SpMat A(s.n_free, s.n_free);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
    s.chol.compute(A);
    s.factored = (s.chol.info() == Eigen::Success);
    s.dt = dt;
}

// One implicit-Euler step. Pins already written into dev.vertex.curr by
// update_constraint() are the Dirichlet targets; free vertices are solved.
inline void step(DataSet &dev, const ParamSet &param) {
    Solver &s = state();
    const int n = static_cast<int>(dev.vertex.curr.size);
    if (n == 0 || dev.shell_face_count == 0) {
        return;
    }
    const double dt = param.dt;
    if (!(dt > 0.0)) {
        return;
    }

    // (Re)build rest-shape operators if the topology changed.
    if (s.n != n || s.faces.empty()) {
        s.n = n;
        rebuild_faces(dev);
        s.factored = false;
    }

    // Fixed set from the constraint. Every `fix` pair is a hard Dirichlet
    // BC at its target `position`, whether or not it is `kinematic` (the
    // kinematic flag only marks animated pins; a static hold is kinematic ==
    // false but still hard-fixed). `pull` pairs are soft and ignored here.
    // Targets come straight from the fix pairs because the emulator's
    // update_constraint only writes kinematic pins into vertex.curr.
    std::vector<char> is_fixed(n, 0);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(n, 3);
    for (unsigned i = 0; i < dev.constraint.fix.size; ++i) {
        const FixPair &p = dev.constraint.fix.data[i];
        if (p.index < static_cast<unsigned>(n)) {
            is_fixed[p.index] = 1;
            target(p.index, 0) = float(p.position[0]);
            target(p.index, 1) = float(p.position[1]);
            target(p.index, 2) = float(p.position[2]);
        }
    }
    const std::size_t sig = hash_fixed(is_fixed);
    if (!s.factored || sig != s.fixed_sig || s.dt != dt ||
        static_cast<int>(s.is_fixed.size()) != n) {
        s.is_fixed = is_fixed;
        s.fixed_sig = sig;
        factor(dt);
    }
    if (!s.factored) {
        return; // factorization failed; leave state untouched
    }

    // Gather positions (x0) and previous positions for velocity. prev holds
    // the start-of-previous-step positions; curr (free entries) hold the
    // start-of-this-step positions because nothing has moved them yet.
    Eigen::MatrixXd x0(n, 3), prev(n, 3);
    for (int v = 0; v < n; ++v) {
        const Vec3f c = dev.vertex.curr.data[v];
        const Vec3f p = dev.vertex.prev.data[v];
        x0(v, 0) = float(c[0]);
        x0(v, 1) = float(c[1]);
        x0(v, 2) = float(c[2]);
        prev(v, 0) = float(p[0]);
        prev(v, 1) = float(p[1]);
        prev(v, 2) = float(p[2]);
    }

    const double prev_dt = (param.prev_dt > 0.0f) ? param.prev_dt : dt;
    const double inv_dt2 = 1.0 / (dt * dt);
    Eigen::RowVector3d g(param.gravity[0], param.gravity[1], param.gravity[2]);

    // Pin the fixed vertices to their targets (prev too, so their velocity
    // is zero and never feeds the free-vertex predictor through coupling).
    for (int v = 0; v < n; ++v) {
        if (is_fixed[v]) {
            x0.row(v) = target.row(v);
            prev.row(v) = target.row(v);
        }
    }

    // Inertial + gravity predictor y (free vertices only are integrated).
    Eigen::MatrixXd y(n, 3);
    for (int v = 0; v < n; ++v) {
        const Eigen::RowVector3d xv = x0.row(v);
        const Eigen::RowVector3d vel = (xv - prev.row(v)) / prev_dt;
        y.row(v) = xv + dt * vel + (dt * dt) * g;
    }

    // Current solution starts at x0; fixed verts pinned to their targets.
    Eigen::MatrixXd x = x0;

    const int n_free = s.n_free;
    const int n_iter = iterations();
    for (int it = 0; it < n_iter; ++it) {
        // RHS: inertia term (constant across iterations) + local ARAP term.
        Eigen::MatrixXd b = Eigen::MatrixXd::Zero(n_free, 3);
        for (int v = 0; v < n; ++v) {
            if (s.is_fixed[v]) {
                continue;
            }
            b.row(s.reduced[v]) += (s.mass[v] * inv_dt2) * y.row(v);
        }
        // Local step + scatter, with Dirichlet condensation of fixed cols.
        for (const FacePre &fp : s.faces) {
            // F = X3 * B, X3 is 3x3 (rows = coord, cols = the 3 verts).
            Mat33d X3;
            for (int a = 0; a < 3; ++a) {
                X3.col(a) = x.row(fp.v[a]).transpose();
            }
            const Mat32d F = X3 * fp.B;
            Eigen::JacobiSVD<Mat32d> svd(F, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
            // Closest 3x2 frame with orthonormal columns (ARAP projection:
            // singular values set to 1). Fixed-size SVD yields a full 3x3 U,
            // so take its first two columns.
            const Mat32d R =
                svd.matrixU().leftCols<2>() * svd.matrixV().transpose();
            // Local RHS contribution: w * B * R(row r)^T, scattered to free.
            for (int a = 0; a < 3; ++a) {
                const int va = fp.v[a];
                if (s.is_fixed[va]) {
                    continue;
                }
                // elastic load: row a of (w * B * R^T) over the 3 coords.
                const Eigen::RowVector2d Ba = fp.B.row(a);
                for (int r = 0; r < 3; ++r) {
                    b(s.reduced[va], r) += fp.w * (Ba.dot(R.row(r)));
                }
            }
            // Condense fixed columns of the elastic stiffness into the RHS:
            // -K_e(a,b) * x_fixed(b) for free a, fixed b.
            for (int bcol = 0; bcol < 3; ++bcol) {
                const int vb = fp.v[bcol];
                if (!s.is_fixed[vb]) {
                    continue;
                }
                for (int a = 0; a < 3; ++a) {
                    const int va = fp.v[a];
                    if (s.is_fixed[va]) {
                        continue;
                    }
                    const double kab =
                        fp.w * fp.B.row(a).dot(fp.B.row(bcol));
                    b.row(s.reduced[va]) -= kab * x.row(vb);
                }
            }
        }
        // Global solve: same factor for all three coordinates.
        Eigen::MatrixXd xf(n_free, 3);
        for (int r = 0; r < 3; ++r) {
            xf.col(r) = s.chol.solve(b.col(r));
        }
        for (int v = 0; v < n; ++v) {
            if (!s.is_fixed[v]) {
                x.row(v) = xf.row(s.reduced[v]);
            }
        }
    }

    // Write back: prev <- start-of-step positions, curr <- solved positions.
    for (int v = 0; v < n; ++v) {
        dev.vertex.prev.data[v] = dev.vertex.curr.data[v];
        Vec3f nc;
        nc[0] = float(static_cast<float>(x(v, 0)));
        nc[1] = float(static_cast<float>(x(v, 1)));
        nc[2] = float(static_cast<float>(x(v, 2)));
        dev.vertex.curr.data[v] = nc;
    }
}

} // namespace pd_arap

#endif
