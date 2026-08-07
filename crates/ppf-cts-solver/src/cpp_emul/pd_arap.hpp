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
//   * Membrane ARAP on shell faces, plus quadratic bending on shell hinges
//     (see HingePre). No tets/rods, no contact/collision, no strain limiting,
//     no plasticity, no damping beyond the implicit integrator's numerical
//     dissipation.
//   * The bending term's absolute stiffness is NOT calibrated against the
//     CUDA solver: that one measures a true dihedral angle with a
//     resolution-independent coefficient, this one is a small-angle quadratic
//     with its own normalization. Its parameter dependence does match, which
//     is what lets the directional `bend-warp` / `bend-weft` stiffnesses be
//     exercised on a machine with no GPU. Do not read absolute drape
//     geometry from it and expect the production solver to agree.
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

// Per shell hinge: the four vertex indices in the mesh's own (i, j, k, l)
// order (shared edge i-j, apexes k and l), the PD constraint weight, the
// linear stencil `c`, the rest configuration it was built from, and the
// stencil's value on that configuration.
//
// The stencil is the quadratic bending model (Bergou et al. 2006): the unique
// affine dependency among the four rest points, so `sum_a c_a x_a` vanishes
// exactly when they are coplanar and grows with the fold otherwise. That makes
// the constraint LINEAR in x, so it assembles into the same prefactored
// Cholesky the membrane term uses.
//
// `rest_vec` is kept as a VECTOR, not as its length. The local step has to
// reproduce the rest curvature with its sign, which is what says "curve this
// way" rather than merely "curve this much": bending carries no contact, so
// there is no reason for this backend to describe a rest angle any more weakly
// than the CUDA kernel's signed (theta - theta_rest) does. Storing only the
// length gives a constraint a flat sheet can satisfy by bending EITHER way,
// and a reference shape then barely moves the result. `rest_p` carries the
// rest points (centred) so the local step can rotate `rest_vec` into the
// current frame.
struct HingePre {
    int v[4];
    double w;
    double c[4];
    Eigen::Vector3d rest_vec;
    Eigen::Vector3d rest_p[4];
};

struct Solver {
    int n = 0;                 // vertex count this factorization was built for
    double dt = 0.0;           // dt this factorization was built for
    std::vector<FacePre> faces;
    std::vector<HingePre> hinges;
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

// Rebuild the per-hinge bending stencils from the rest pose. Called beside
// rebuild_faces, and like it this bakes the material parameters in (the
// membrane does the same with `mu`), so a parameter change takes effect at the
// next initialize rather than mid-run.
//
// The stencil is derived rather than looked up. Writing the rest points
// relative to the shared-edge endpoint j (every quantity below is a DIFFERENCE
// of positions, so the stencil is translation-invariant by construction), the
// affine dependency
// `sum_a c_a x_a = 0` with `sum_a c_a = 0` has apex coefficients proportional
// to the OPPOSITE triangle's area:
//
//   c_k = -2 * area(i, j, l),   c_l = -2 * area(i, j, k)
//
// and the shared-edge pair then follows from the dependency itself. Scaling by
// 2*(A1 + A2) leaves `c_i + c_j = 1` and `c_k + c_l = -1`, so `c` is
// dimensionless and mesh-scale free.
inline void rebuild_hinges(const DataSet &d) {
    Solver &s = state();
    s.hinges.clear();
    const unsigned nhinge = d.mesh.mesh.hinge.size;
    s.hinges.reserve(nhinge);
    for (unsigned h = 0; h < nhinge; ++h) {
        const HingeProp &hp = d.prop.hinge.data[h];
        // A fully pinned or collider hinge carries no bending energy, matching
        // the face gate in the CUDA path.
        if (hp.fixed || hp.collider) {
            continue;
        }
        const HingeParam &par = d.param_arrays.hinge.data[hp.param_index];
        if (!(par.bend > 0.0f)) {
            continue;
        }
        const Vec4u hg = d.mesh.mesh.hinge.data[h];
        const Vec3f origin = d.vertex.curr.data[hg[1]];
        const Vec3f edge = d.vertex.curr.data[hg[0]] - origin;
        const Vec3f to_k = d.vertex.curr.data[hg[2]] - origin;
        const Vec3f to_l = d.vertex.curr.data[hg[3]] - origin;
        const Eigen::Vector3d a(edge[0], edge[1], edge[2]);
        const Eigen::Vector3d k(to_k[0], to_k[1], to_k[2]);
        const Eigen::Vector3d l(to_l[0], to_l[1], to_l[2]);
        const double edge_sq = a.squaredNorm();
        const double twice_a1 = a.cross(k).norm();
        const double twice_a2 = a.cross(l).norm();
        const double scale = twice_a1 + twice_a2;
        if (!(edge_sq > 0.0) || !(scale > 0.0)) {
            continue; // degenerate rest hinge carries no bending
        }
        HingePre pre;
        pre.v[0] = static_cast<int>(hg[0]);
        pre.v[1] = static_cast<int>(hg[1]);
        pre.v[2] = static_cast<int>(hg[2]);
        pre.v[3] = static_cast<int>(hg[3]);
        const double ck = -twice_a2 / scale;
        const double cl = -twice_a1 / scale;
        // Solve the dependency for the shared-edge endpoints: with j as the
        // origin the j term drops out (sum c = 0), leaving c_i along `a`.
        const Eigen::Vector3d apex = ck * k + cl * l;
        const double ci = -apex.dot(a) / edge_sq;
        pre.c[0] = ci;
        pre.c[1] = -(ci + ck + cl);
        pre.c[2] = ck;
        pre.c[3] = cl;
        // Rest curvature to preserve. This has to honor HingeProp::rest_angle,
        // which is where a bending REFERENCE lands: builder.rs computes it from
        // the reference shape (or from the initial pose when the group asks for
        // that), and the CUDA kernel consumes it directly as the angle the
        // dihedral energy is measured against. The reference VERTICES never
        // reach the backend, only that angle, so the rest configuration is
        // reconstructed here by rotating apex l about the shared edge from the
        // initial pose's angle to the target. Reading the residual off the
        // initial pose instead would silently ignore the reference and leave
        // this backend disagreeing with the CUDA one on exactly the scenes a
        // reference exists to describe.
        {
            // Mirrors signed_dihedral_angle in builder.rs, which is what
            // produced HingeProp::rest_angle, under that function's remap
            // (v0, v1, v2, v3) = (k, j, i, l) with j at the origin:
            //   n1 = (j - k) x (i - k) =  a x k
            //   n2 = (i - l) x (j - l) =  l x a
            // The second cross is l x a and NOT a x l. The apexes lie on
            // OPPOSITE sides of the shared edge, so the flipped spelling makes
            // a flat hinge read as pi instead of 0, which then rotates every
            // rest configuration by half a turn and gives a flat sheet a
            // spurious rest curvature.
            const Eigen::Vector3d n1 = a.cross(k);
            const Eigen::Vector3d n2 = l.cross(a);
            const double n1n = n1.norm(), n2n = n2.norm();
            double theta0 = 0.0;
            if (n1n > 0.0 && n2n > 0.0) {
                const double c01 = std::max(-1.0, std::min(1.0,
                                            n1.dot(n2) / (n1n * n2n)));
                theta0 = std::acos(c01);
                // Signed, about the shared edge, so a fold and its mirror do
                // not collapse onto the same rotation below.
                if (n1.cross(n2).dot(a) < 0.0) {
                    theta0 = -theta0;
                }
            }
            const double delta = static_cast<double>(hp.rest_angle) - theta0;
            Eigen::Vector3d l_rest = l;
            if (std::abs(delta) > 1e-12 && edge_sq > 0.0) {
                const Eigen::Vector3d axis = a / std::sqrt(edge_sq);
                const double cd = std::cos(delta), sd = std::sin(delta);
                // Rodrigues about the shared edge through the origin (j).
                l_rest = l * cd + axis.cross(l) * sd +
                         axis * (axis.dot(l)) * (1.0 - cd);
            }
            // Keep the rest configuration itself, centred, plus the stencil's
            // SIGNED value on it. The local step rotates that vector into the
            // current frame rather than reusing the current bend direction.
            const Eigen::Vector3d rp[4] = {a, Eigen::Vector3d::Zero(), k,
                                           l_rest};
            Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
            for (int q = 0; q < 4; ++q) {
                centroid += rp[q];
            }
            centroid *= 0.25;
            for (int q = 0; q < 4; ++q) {
                pre.rest_p[q] = rp[q] - centroid;
            }
            // sum c = 0, so this is translation invariant and the centring
            // above does not change it.
            pre.rest_vec = ci * a + ck * k + cl * l_rest;
        }
        // Isotropic plus directional stiffness, from the same shared helper
        // the CUDA kernel uses so the two cannot drift. The directional part
        // depends on the edge's UV DIRECTION alone, which is scale free, so it
        // is invariant under refinement and cannot disturb the scaling below.
        const double bend_dir = static_cast<double>(hinge_bend_directional(
            par.bend, par.bend_warp, par.bend_weft, hp.uv_edge_sin2));
        // Areal density (mass per unit area) averaged over the hinge's four
        // vertices, matching embed_hinge_force_hessian so `bend` alone sets
        // the bent shape and density stays a free knob.
        double areal_density = 0.0;
        int counted = 0;
        for (int q = 0; q < 4; ++q) {
            const double va = d.prop.vertex.data[hg[q]].area;
            if (va > 0.0) {
                areal_density += d.prop.vertex.data[hg[q]].mass / va;
                ++counted;
            }
        }
        areal_density = counted > 0 ? areal_density / counted : 1.0;
        // RESOLUTION INDEPENDENCE. `c` is dimensionless, so for a surface of
        // curvature kappa meshed at spacing h the stencil reports
        // |sum c_a x_a| ~ kappa * h^2, and there are ~area/h^2 hinges. The
        // per-hinge weight must therefore go as 1/h^2 for the total energy to
        // converge to the continuum integral of B*kappa^2 dA rather than drift
        // with the mesh: taking w ~ 1/(A1 + A2) gives
        // (1/h^2) * (kappa h^2)^2 * (area/h^2) = kappa^2 * area, independent of
        // h. Weighting by (A1 + A2) instead -- the obvious mirror of the
        // membrane's `mu * area` -- scales as h^4 and makes bending VANISH
        // under refinement, which is the trap this comment exists to prevent.
        const double hinge_area = 0.5 * scale; // (A1 + A2)
        pre.w = bend_dir * areal_density / hinge_area;
        if (!(pre.w > 0.0) || !std::isfinite(pre.w)) {
            continue;
        }
        s.hinges.push_back(pre);
    }
}

inline void lock_basis(const Eigen::Vector3d &axis, Eigen::Vector3d &b0,
                       Eigen::Vector3d &b1) {
    const Eigen::Vector3d reference =
        std::abs(axis.z()) < 0.9 ? Eigen::Vector3d::UnitZ()
                                : Eigen::Vector3d::UnitX();
    b0 = axis.cross(reference).normalized();
    b1 = axis.cross(b0).normalized();
}

inline void factor(const DataSet &d, double dt) {
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
    trips.reserve(s.faces.size() * 9 + s.hinges.size() * 16 + s.n_free);
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
    // Bending: the stencil is linear in x, so its Hessian w * c c^T is
    // constant and joins the same prefactored matrix. c c^T is PSD for any
    // c and w is positive, so this cannot spoil the Cholesky.
    for (const HingePre &hp : s.hinges) {
        for (int a = 0; a < 4; ++a) {
            const int va = hp.v[a];
            if (s.is_fixed[va]) {
                continue;
            }
            for (int b = 0; b < 4; ++b) {
                const int vb = hp.v[b];
                if (s.is_fixed[vb]) {
                    continue; // fixed columns are condensed into the RHS
                }
                trips.emplace_back(s.reduced[va], s.reduced[vb],
                                   hp.w * hp.c[a] * hp.c[b]);
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

inline bool lock_axis_enabled(const Vec3f &axis) {
    return axis[0] != 0.0f || axis[1] != 0.0f || axis[2] != 0.0f;
}

inline bool rotation_lock_mode_valid(unsigned mode) {
    return mode == ROTATION_LOCK_ALLOW_ONLY ||
           mode == ROTATION_LOCK_PROHIBIT_AXIS;
}

inline Eigen::Vector3d lock_axis(const Vec3f &axis, unsigned dmap_index,
                                 const char *kind) {
    const Eigen::Vector3d a(axis[0], axis[1], axis[2]);
    const double norm = a.norm();
    if (!a.array().isFinite().all() || !(norm > 0.0) ||
        !std::isfinite(norm)) {
        fprintf(stderr,
                "PPF FATAL: emulated %s lock displacement group %u has an "
                "invalid axis.\n",
                kind, dmap_index);
        std::abort();
    }
    return a / norm;
}

struct RotationFrame {
    unsigned lock_index;
    unsigned mode;
    Eigen::Vector3d axis;
    Eigen::Vector3d com;
    Eigen::Matrix3d inverse_inertia;
};

struct DynamicConstraints {
    Eigen::MatrixXd C;
    Eigen::VectorXd rhs;
    std::vector<RotationFrame> rotation_frames;
};

struct ConstraintLayout {
    int n;
    int n_free;
    const std::vector<char> &is_fixed;
    const std::vector<int> &reduced;
};

inline RotationFrame make_rotation_frame(const DataSet &d,
                                         const Eigen::MatrixXd &reference,
                                         unsigned lock_index) {
    const TranslationLock &lock = d.translation_lock.data[lock_index];
    double total_mass = 0.0;
    Eigen::Vector3d com_numerator = Eigen::Vector3d::Zero();
    for (int v = 0; v < static_cast<int>(reference.rows()); ++v) {
        if (d.translation_lock_index.data[v] != lock_index) {
            continue;
        }
        const double mass = d.prop.vertex.data[v].mass;
        total_mass += mass;
        com_numerator += mass * reference.row(v).transpose();
    }
    if (!(total_mass > 0.0) || !std::isfinite(total_mass) ||
        !com_numerator.array().isFinite().all()) {
        fprintf(stderr,
                "PPF FATAL: emulated rotation lock displacement group %u "
                "has no finite positive-mass frame.\n",
                lock.dmap_index);
        std::abort();
    }

    const Eigen::Vector3d com = com_numerator / total_mass;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    for (int v = 0; v < static_cast<int>(reference.rows()); ++v) {
        if (d.translation_lock_index.data[v] != lock_index) {
            continue;
        }
        const double mass = d.prop.vertex.data[v].mass;
        const Eigen::Vector3d r = reference.row(v).transpose() - com;
        inertia += mass * (r.squaredNorm() * Eigen::Matrix3d::Identity() -
                           r * r.transpose());
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(inertia);
    if (eig.info() != Eigen::Success ||
        !eig.eigenvalues().array().isFinite().all()) {
        fprintf(stderr,
                "PPF FATAL: emulated rotation lock displacement group %u "
                "has a non-finite inertia frame.\n",
                lock.dmap_index);
        std::abort();
    }
    const double scale = eig.eigenvalues().cwiseAbs().maxCoeff();
    const double rank_floor =
        128.0 * std::numeric_limits<float>::epsilon() * scale;
    if (!(scale > 0.0) || !std::isfinite(scale) ||
        !(eig.eigenvalues().minCoeff() > rank_floor)) {
        fprintf(stderr,
                "PPF FATAL: emulated rotation lock displacement group %u "
                "has singular or float32-unresolvable inertia.\n",
                lock.dmap_index);
        std::abort();
    }
    RotationFrame frame;
    frame.lock_index = lock_index;
    frame.mode = lock.rotation_mode;
    frame.axis = lock_axis(lock.rotation_axis, lock.dmap_index, "rotation");
    frame.com = com;
    frame.inverse_inertia =
        eig.eigenvectors() * eig.eigenvalues().cwiseInverse().asDiagonal() *
        eig.eigenvectors().transpose();
    return frame;
}

// Build C_free and h - C_fixed p for the current local-global iteration.
// Translation rows keep the initial perpendicular COM position. Rotation rows
// keep the best-fit infinitesimal angular increment of this iteration on the
// requested axis. Allow-only contributes two tangent rows, while
// prohibit-axis contributes one row along the axis. These rows must be rebuilt
// because their inertia frame and per-vertex coefficients depend on the
// current iterate.
inline DynamicConstraints build_dynamic_constraints(
    const DataSet &d, const Eigen::MatrixXd &reference,
    const Eigen::MatrixXd &fixed_values, const ConstraintLayout &layout) {
    DynamicConstraints out;
    out.C.resize(0, 3 * layout.n_free);
    out.rhs.resize(0);
    if (d.translation_lock.size == 0) {
        return out;
    }
    if (!reference.array().isFinite().all() ||
        !fixed_values.array().isFinite().all()) {
        fprintf(stderr,
                "PPF FATAL: emulated aggregate lock received non-finite "
                "local-global positions.\n");
        std::abort();
    }

    std::vector<Eigen::RowVectorXd> rows;
    std::vector<double> rhs;
    out.rotation_frames.reserve(d.translation_lock.size);
    for (unsigned li = 0; li < d.translation_lock.size; ++li) {
        const TranslationLock &lock = d.translation_lock.data[li];
        if (!rotation_lock_mode_valid(lock.rotation_mode)) {
            fprintf(stderr,
                    "PPF FATAL: emulated rotation lock displacement group "
                    "%u has invalid mode %u.\n",
                    lock.dmap_index, lock.rotation_mode);
            std::abort();
        }
        const bool translation_enabled = lock_axis_enabled(lock.axis);
        const bool rotation_enabled = lock_axis_enabled(lock.rotation_axis);
        if (!translation_enabled && !rotation_enabled) {
            fprintf(stderr,
                    "PPF FATAL: emulated aggregate lock displacement group "
                    "%u has no enabled component.\n",
                    lock.dmap_index);
            std::abort();
        }

        auto append_row = [&](const Eigen::Vector3d &basis,
                              const RotationFrame *rotation) {
            Eigen::RowVectorXd row =
                Eigen::RowVectorXd::Zero(3 * layout.n_free);
            double target = 0.0;
            double fixed = 0.0;
            for (int v = 0; v < layout.n; ++v) {
                if (d.translation_lock_index.data[v] != li) {
                    continue;
                }
                const double mass = d.prop.vertex.data[v].mass;
                Eigen::Vector3d coefficient;
                if (rotation) {
                    const Eigen::Vector3d r =
                        reference.row(v).transpose() - rotation->com;
                    coefficient =
                        mass * (rotation->inverse_inertia * basis).cross(r);
                    target += coefficient.dot(reference.row(v).transpose());
                } else {
                    coefficient = mass * basis;
                    const Vec3f initial = d.translation_lock_initial.data[v];
                    const Eigen::Vector3d initial_position(
                        initial[0], initial[1], initial[2]);
                    target += coefficient.dot(initial_position);
                }
                if (layout.is_fixed[v]) {
                    fixed += coefficient.dot(fixed_values.row(v).transpose());
                } else {
                    row.segment<3>(3 * layout.reduced[v]) =
                        coefficient.transpose();
                }
            }
            rows.push_back(std::move(row));
            rhs.push_back(target - fixed);
        };

        if (translation_enabled) {
            const Eigen::Vector3d axis =
                lock_axis(lock.axis, lock.dmap_index, "translation");
            Eigen::Vector3d b0, b1;
            lock_basis(axis, b0, b1);
            append_row(b0, nullptr);
            append_row(b1, nullptr);
        }
        if (rotation_enabled) {
            out.rotation_frames.push_back(
                make_rotation_frame(d, reference, li));
            const RotationFrame &frame = out.rotation_frames.back();
            if (frame.mode == ROTATION_LOCK_PROHIBIT_AXIS) {
                append_row(frame.axis, &frame);
            } else {
                Eigen::Vector3d b0, b1;
                lock_basis(frame.axis, b0, b1);
                append_row(b0, &frame);
                append_row(b1, &frame);
            }
        }
    }

    out.C.resize(static_cast<int>(rows.size()), 3 * layout.n_free);
    out.rhs.resize(static_cast<int>(rhs.size()));
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
        out.C.row(row) = rows[row];
        out.rhs[row] = rhs[row];
    }
    return out;
}

inline void check_rotation_tangent(const DataSet &d,
                                   const Eigen::MatrixXd &reference,
                                   const Eigen::MatrixXd &candidate,
                                   const DynamicConstraints &constraints) {
    constexpr double eps = std::numeric_limits<float>::epsilon();
    for (const RotationFrame &frame : constraints.rotation_frames) {
        Eigen::Vector3d torque = Eigen::Vector3d::Zero();
        for (int v = 0; v < static_cast<int>(reference.rows()); ++v) {
            if (d.translation_lock_index.data[v] != frame.lock_index) {
                continue;
            }
            const double mass = d.prop.vertex.data[v].mass;
            const Eigen::Vector3d r =
                reference.row(v).transpose() - frame.com;
            const Eigen::Vector3d dx =
                candidate.row(v).transpose() - reference.row(v).transpose();
            torque += mass * r.cross(dx);
        }
        const Eigen::Vector3d omega = frame.inverse_inertia * torque;
        const double forbidden = frame.mode == ROTATION_LOCK_PROHIBIT_AXIS
            ? std::abs(frame.axis.dot(omega))
            : (omega - frame.axis * frame.axis.dot(omega)).norm();
        const double inverse_scale =
            frame.inverse_inertia.cwiseAbs().maxCoeff();
        const double bound = 4096.0 * eps *
            std::max(1.0, omega.norm() + torque.norm() * inverse_scale);
        if (!omega.array().isFinite().all() ||
            !std::isfinite(forbidden) || forbidden > bound) {
            fprintf(stderr,
                    "PPF FATAL: emulated rotation lock group %u violated "
                    "its finite best-fit angular invariant (forbidden "
                    "increment %.6e, bound %.6e).\n",
                    d.translation_lock.data[frame.lock_index].dmap_index,
                    forbidden, bound);
            std::abort();
        }
    }
}

inline Eigen::Vector3d translation_lock_residual(const DataSet &d,
                                                 const Eigen::MatrixXd &x,
                                                 unsigned lock_index) {
    const TranslationLock &lock = d.translation_lock.data[lock_index];
    if (!lock_axis_enabled(lock.axis)) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (int v = 0; v < static_cast<int>(d.vertex.curr.size); ++v) {
        if (d.translation_lock_index.data[v] != lock_index) {
            continue;
        }
        const double mass = d.prop.vertex.data[v].mass;
        const Vec3f initial = d.translation_lock_initial.data[v];
        sum.x() += mass * (x(v, 0) - initial[0]);
        sum.y() += mass * (x(v, 1) - initial[1]);
        sum.z() += mass * (x(v, 2) - initial[2]);
    }
    const Eigen::Vector3d axis =
        lock_axis(lock.axis, lock.dmap_index, "translation");
    return sum - axis * sum.dot(axis);
}

// Project a free-coordinate vector with a caller-supplied A^-1 C^T response.
// Both PD and the emulator's diagonal-mass point-cloud solve use this exact
// Schur complement step, so neither path needs a post-step lock correction.
inline void project_with_schur(const DynamicConstraints &constraints,
                               const Eigen::MatrixXd &response,
                               Eigen::VectorXd &free_x) {
    const int nrow = constraints.C.rows();
    const Eigen::VectorXd residual = constraints.C * free_x - constraints.rhs;
    Eigen::MatrixXd schur = constraints.C * response;
    schur = 0.5 * (schur + schur.transpose());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        schur, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (!singular.array().isFinite().all()) {
        fprintf(stderr,
                "PPF FATAL: emulated aggregate lock Schur complement is "
                "non-finite.\n");
        std::abort();
    }
    const double largest = singular.size() ? singular.maxCoeff() : 0.0;
    const double rank_floor =
        256.0 * std::numeric_limits<double>::epsilon() * largest;
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(nrow);
    for (int k = 0; k < singular.size(); ++k) {
        if (singular[k] > rank_floor) {
            lambda += svd.matrixV().col(k) *
                (svd.matrixU().col(k).dot(residual) / singular[k]);
        }
    }
    const Eigen::VectorXd unresolved = residual - schur * lambda;
    const double compatibility_scale =
        std::max(1.0, std::max(constraints.rhs.norm(), residual.norm()));
    const double compatibility_bound =
        4096.0 * std::numeric_limits<double>::epsilon() * compatibility_scale;
    if (!lambda.array().isFinite().all() ||
        !unresolved.array().isFinite().all() ||
        unresolved.norm() > compatibility_bound) {
        fprintf(stderr,
                "PPF FATAL: emulated aggregate lock is infeasible: exact "
                "fixed vertices leave a residual %.6e outside the free "
                "constraint space (bound %.6e).\n",
                unresolved.norm(), compatibility_bound);
        std::abort();
    }
    free_x -= response * lambda;
    const Eigen::VectorXd final_residual =
        constraints.C * free_x - constraints.rhs;
    const double final_bound =
        8192.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, constraints.rhs.norm());
    if (!free_x.array().isFinite().all() ||
        !final_residual.array().isFinite().all() ||
        final_residual.norm() > final_bound) {
        fprintf(stderr,
                "PPF FATAL: emulated aggregate lock projection left "
                "constraint residual %.6e (bound %.6e).\n",
                final_residual.norm(), final_bound);
        std::abort();
    }
}

// Apply the exact A-metric projection to an unconstrained global-step result:
//
//   x = x_u - A^-1 C^T (C A^-1 C^T)^+ (C x_u - h).
//
// The small Schur complement is rebuilt inside each local-global iteration.
// This is part of the global solve, not a post-step position correction.
inline void constrain_global_solution(const DataSet &d,
                                      const Eigen::MatrixXd &reference,
                                      const Eigen::MatrixXd &fixed_values,
                                      Eigen::MatrixXd &xf) {
    Solver &s = state();
    if (d.translation_lock.size == 0) {
        return;
    }
    const ConstraintLayout layout{
        s.n, s.n_free, s.is_fixed, s.reduced};
    const DynamicConstraints constraints =
        build_dynamic_constraints(d, reference, fixed_values, layout);
    const int nrow = constraints.C.rows();
    if (nrow == 0) {
        return;
    }

    Eigen::VectorXd free_x(3 * s.n_free);
    for (int v = 0; v < s.n_free; ++v) {
        free_x.segment<3>(3 * v) = xf.row(v).transpose();
    }
    Eigen::MatrixXd response = Eigen::MatrixXd::Zero(3 * s.n_free, nrow);
    for (int row = 0; row < nrow; ++row) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(s.n_free);
            for (int v = 0; v < s.n_free; ++v) {
                rhs[v] = constraints.C(row, 3 * v + coordinate);
            }
            const Eigen::VectorXd solved = s.chol.solve(rhs);
            if (s.chol.info() != Eigen::Success ||
                !solved.array().isFinite().all()) {
                fprintf(stderr,
                        "PPF FATAL: emulated aggregate lock A-inverse "
                        "application failed.\n");
                std::abort();
            }
            for (int v = 0; v < s.n_free; ++v) {
                response(3 * v + coordinate, row) = solved[v];
            }
        }
    }
    project_with_schur(constraints, response, free_x);
    for (int v = 0; v < s.n_free; ++v) {
        xf.row(v) = free_x.segment<3>(3 * v).transpose();
    }

    Eigen::MatrixXd candidate = fixed_values;
    for (int v = 0; v < s.n; ++v) {
        if (!s.is_fixed[v]) {
            candidate.row(v) = xf.row(s.reduced[v]);
        }
    }
    check_rotation_tangent(d, reference, candidate, constraints);
}

// One implicit-Euler step. Pins already written into dev.vertex.curr by
// update_constraint() are the Dirichlet targets; free vertices are solved.
inline bool step(DataSet &dev, const ParamSet &param) {
    Solver &s = state();
    const int n = static_cast<int>(dev.vertex.curr.size);
    if (n == 0 || dev.shell_face_count == 0) {
        return false;
    }
    const double dt = param.dt;
    if (!(dt > 0.0)) {
        return false;
    }

    // (Re)build rest-shape operators if the topology changed.
    if (s.n != n || s.faces.empty()) {
        s.n = n;
        rebuild_faces(dev);
        rebuild_hinges(dev);
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
        factor(dev, dt);
    }
    if (!s.factored) {
        return false; // factorization failed; leave state untouched
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
    // Keep the pre-pin geometry for the first constrained global solve.
    // A moved exact pin is an affine prescribed increment relative to this
    // state, so its contribution must be included in the rotation rows rather
    // than hidden by replacing x0 before the constraint operator is built.
    Eigen::MatrixXd pre_pin_x = x0;
    for (unsigned i = 0; i < dev.constraint.fix.size; ++i) {
        const FixPair &pin = dev.constraint.fix.data[i];
        if (pin.kinematic && pin.index < static_cast<unsigned>(n)) {
            // update_constraint preserved this vertex's pre-target position in
            // prev before assigning the kinematic target to curr.
            pre_pin_x.row(pin.index) = prev.row(pin.index);
        }
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
        // Bending local step + scatter, same Dirichlet condensation.
        for (const HingePre &hp : s.hinges) {
            Eigen::RowVector3d cx = Eigen::RowVector3d::Zero();
            for (int a = 0; a < 4; ++a) {
                cx += hp.c[a] * x.row(hp.v[a]);
            }
            // Local step: carry the rest curvature into the current frame with
            // its SIGN, by best-fitting the rest hinge onto the current one and
            // rotating `rest_vec` by that rotation. This is the same shape that
            // the membrane's ARAP step above uses (SVD, closest rotation), and
            // it is what makes a rest angle mean "curve THIS way".
            //
            // Reusing the current bend direction instead (scaling cx to the
            // rest length) is rotation invariant too and looks equivalent, but
            // it is strictly weaker: a flat sheet can then satisfy the
            // constraint by bending either way, so a bending REFERENCE barely
            // moves the result and this backend disagrees with the CUDA
            // kernel's signed (theta - theta_rest) on exactly the scenes a
            // reference exists to describe. Measured on the reference
            // scenario, that spelling moved the drape under 1%.
            //
            // A flat rest hinge has rest_vec == 0, so the target is the origin,
            // the rotation is irrelevant and the term stays purely quadratic.
            Eigen::RowVector3d p = Eigen::RowVector3d::Zero();
            if (hp.rest_vec.squaredNorm() > 0.0) {
                Eigen::RowVector3d cur_c = Eigen::RowVector3d::Zero();
                for (int a = 0; a < 4; ++a) {
                    cur_c += x.row(hp.v[a]);
                }
                cur_c *= 0.25;
                Mat33d H = Mat33d::Zero();
                for (int a = 0; a < 4; ++a) {
                    H += hp.rest_p[a] *
                         (x.row(hp.v[a]) - cur_c);   // 3x1 * 1x3
                }
                Eigen::JacobiSVD<Mat33d> svd(
                    H, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Mat33d R = svd.matrixV() * svd.matrixU().transpose();
                if (R.determinant() < 0.0) {
                    // Reflection: flip the least-significant singular axis so
                    // the fit stays a rotation rather than a mirror, which
                    // would invert the rest curvature's sign.
                    Mat33d V = svd.matrixV();
                    V.col(2) *= -1.0;
                    R = V * svd.matrixU().transpose();
                }
                p = (R * hp.rest_vec).transpose();
            }
            for (int a = 0; a < 4; ++a) {
                const int va = hp.v[a];
                if (s.is_fixed[va]) {
                    continue;
                }
                b.row(s.reduced[va]) += (hp.w * hp.c[a]) * p;
            }
            for (int bcol = 0; bcol < 4; ++bcol) {
                const int vb = hp.v[bcol];
                if (!s.is_fixed[vb]) {
                    continue;
                }
                for (int a = 0; a < 4; ++a) {
                    const int va = hp.v[a];
                    if (s.is_fixed[va]) {
                        continue;
                    }
                    b.row(s.reduced[va]) -=
                        (hp.w * hp.c[a] * hp.c[bcol]) * x.row(vb);
                }
            }
        }
        // Global solve: same factor for all three coordinates.
        Eigen::MatrixXd xf(n_free, 3);
        for (int r = 0; r < 3; ++r) {
            xf.col(r) = s.chol.solve(b.col(r));
        }
        const Eigen::MatrixXd &lock_reference =
            it == 0 ? pre_pin_x : x;
        constrain_global_solution(dev, lock_reference, x, xf);
        for (int v = 0; v < n; ++v) {
            if (!s.is_fixed[v]) {
                x.row(v) = xf.row(s.reduced[v]);
            }
        }
    }

    // The global solve must produce a feasible state before writeback. This is
    // an invariant check, never a corrective projection.
    for (unsigned li = 0; li < dev.translation_lock.size; ++li) {
        const double residual = translation_lock_residual(dev, x, li).norm() /
                                dev.translation_lock.data[li].total_mass;
        if (!std::isfinite(residual) || residual > 1e-9) {
            fprintf(stderr,
                    "PPF FATAL: emulated translation lock group %u left "
                    "perpendicular COM drift %.6e after the constrained "
                    "global solve.\n",
                    dev.translation_lock.data[li].dmap_index, residual);
            std::abort();
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
    return true;
}

} // namespace pd_arap

#endif
