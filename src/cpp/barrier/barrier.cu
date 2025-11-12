// File: barrier.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "barrier.hpp"
#include "cubic.hpp"
#include "logarithm.hpp"
#include "quadratic.hpp"

namespace barrier {

__device__ float energy(float g, float ghat, float offset, Barrier barrier) {
    if (barrier == Barrier::Cubic) {
        return cubic::energy(g, ghat, offset);
    } else if (barrier == Barrier::Quad) {
        return quadratic::energy(g, ghat, offset);
    } else if (barrier == Barrier::Log) {
        return logarithm::energy(g, ghat, offset);
    } else {
        return 0.0f;
    }
}

__device__ float gradient(float g, float ghat, float offset, Barrier barrier) {
    if (barrier == Barrier::Cubic) {
        return cubic::gradient(g, ghat, offset);
    } else if (barrier == Barrier::Quad) {
        return quadratic::gradient(g, ghat, offset);
    } else if (barrier == Barrier::Log) {
        return logarithm::gradient(g, ghat, offset);
    } else {
        return 0.0f;
    }
}

__device__ float curvature(float g, float ghat, float offset, Barrier barrier) {
    if (barrier == Barrier::Cubic) {
        return cubic::curvature(g, ghat, offset);
    } else if (barrier == Barrier::Quad) {
        return quadratic::curvature(g, ghat, offset);
    } else if (barrier == Barrier::Log) {
        return logarithm::curvature(g, ghat, offset);
    } else {
        return 0.0f;
    }
}

template <unsigned N>
__device__ float
compute_stiffness(const Proximity<N> &prox, const SVecf<N> &mass,
                  const FixedCSRMat &hess, const Vec3f &e, float ghat,
                  float offset, const ParamSet &param) {
    SMatf<N * 3, N * 3> local_hess = SMatf<N * 3, N * 3>::Zero();
    float norm = e.norm();
    float g = norm - offset;
    assert(g > 0.0f);
    float sqr_x = g * g;
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            Mat3x3f val = Mat3x3f::Zero();
            val += hess(prox.index[ii], prox.index[jj]);
            if (ii == jj) {
                val += (mass[ii] / sqr_x) * Mat3x3f::Identity();
            }
            local_hess.template block<3, 3>(3 * ii, 3 * jj) = val;
        }
    }
    SVecf<N * 3> w = SVecf<N * 3>::Zero();
    for (unsigned ii = 0; ii < N; ++ii) {
        float val = prox.value[ii];
        Map<Vec3f>(w.data() + 3 * ii) = val * e;
    }
    w.normalize();
    return (local_hess * w).dot(w);
}

template __device__ float
compute_stiffness<4>(const Proximity<4> &prox, const SVecf<4> &mass,
                     const FixedCSRMat &hess, const Vec3f &e, float ghat,
                     float offset, const ParamSet &param);

template __device__ float
compute_stiffness<3>(const Proximity<3> &prox, const SVecf<3> &mass,
                     const FixedCSRMat &hess, const Vec3f &e, float ghat,
                     float offset, const ParamSet &param);

template __device__ float
compute_stiffness<2>(const Proximity<2> &prox, const SVecf<2> &mass,
                     const FixedCSRMat &hess, const Vec3f &e, float ghat,
                     float offset, const ParamSet &param);

__device__ Vec3f compute_edge_gradient(const Vec3f &e, float ghat, float offset,
                                       Barrier barrier) {
    float g = e.norm();
    Vec3f n = e / g;
    return n * gradient(g, ghat, offset, barrier);
}

__device__ Mat3x3f compute_edge_hessian(const Vec3f &e, float ghat,
                                        float offset, Barrier barrier) {
    float c = curvature(e.norm(), ghat, offset, barrier);
    return c * e * e.transpose() / e.squaredNorm();
}

__device__ DiffTable2 compute_strainlimiting_diff_table(const Vec2f &a,
                                                        float ghat,
                                                        Barrier barrier) {
    DiffTable2 table;
    table.d2ed2a = Mat2x2f::Zero();
    table.deda = Vec2f::Zero();
    for (int i = 0; i < 2; ++i) {
        float g = a[i];
        if (g > 0.0f) {
            float y = ghat - g;
            table.deda[i] = -gradient(y, ghat, 0.0f, barrier);
            table.d2ed2a(i, i) = curvature(y, ghat, 0.0f, barrier);
        }
    }
    return table;
}

__device__ float strainlimiting_energy(const Vec2f &a, float ghat,
                                       Barrier barrier) {
    float result(0.0f);
    for (int i = 0; i < 2; ++i) {
        float g = a[i];
        if (g > 0.0f) {
            float y = ghat - g;
            result += energy(y, ghat, 0.0f, barrier);
        }
    }
    return result;
}

} // namespace barrier
