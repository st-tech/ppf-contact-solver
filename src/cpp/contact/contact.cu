// File: contact.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../barrier/barrier.hpp"
#include "../buffer/buffer.hpp"
#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../energy/model/fix.hpp"
#include "../energy/model/friction.hpp"
#include "../energy/model/push.hpp"
#include "../kernels/reduce.hpp"
#include "../kernels/vec_ops.hpp"
#include "../main/cuda_utils.hpp"
#include "../simplelog/SimpleLog.h"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "aabb.hpp"
#include "accd.hpp"
#include "contact.hpp"
#include "distance.hpp"
#include <cassert>

namespace contact {

namespace storage {
Vec<AABB> face_aabb, edge_aabb, vertex_aabb;
Vec<AABB> collision_mesh_face_aabb, collision_mesh_edge_aabb;
} // namespace storage

static unsigned max_of_two(unsigned a, unsigned b) { return a > b ? a : b; }
__device__ inline float sqr(float x) { return x * x; }

void initialize(const DataSet &data, const ParamSet &param) {
    BVHSet bvhset = data.bvh;

    // Initialize AABB buffers (kept in storage namespace)
    storage::face_aabb =
        Vec<AABB>::alloc(bvhset.face.node.size, param.bvh_alloc_factor);
    storage::edge_aabb =
        Vec<AABB>::alloc(bvhset.edge.node.size, param.bvh_alloc_factor);
    storage::vertex_aabb =
        Vec<AABB>::alloc(bvhset.vertex.node.size, param.bvh_alloc_factor);
    storage::collision_mesh_face_aabb =
        Vec<AABB>::alloc(data.constraint.mesh.face_bvh.node.size);
    storage::collision_mesh_edge_aabb =
        Vec<AABB>::alloc(data.constraint.mesh.edge_bvh.node.size);

    // Temporary buffers are now allocated on demand from the buffer pool
}

void resize_aabb(const BVHSet &bvh) {
    storage::face_aabb.resize(bvh.face.node.size);
    storage::edge_aabb.resize(bvh.edge.node.size);
}

__device__ void update_face_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                                 const BVH &bvh, Vec<AABB> &aabb,
                                 const Vec<Vec3u> &face,
                                 const Vec<FaceProp> &prop,
                                 const Vec<FaceParam> &face_params,
                                 unsigned level, const ParamSet &param,
                                 float extrapolate, unsigned i) {
    unsigned index = bvh.level(level, i);
    Vec2u node = bvh.node[index];
    if (node[1] == 0) {
        unsigned leaf = node[0] - 1;
        const FaceParam &fparam = face_params[prop[leaf].param_index];
        float ext_eps = 0.5f * fparam.ghat + fparam.offset;
        Vec3f y00 = x0[face[leaf][0]];
        Vec3f y01 = x0[face[leaf][1]];
        Vec3f y02 = x0[face[leaf][2]];
        Vec3f y10 = x1[face[leaf][0]];
        Vec3f y11 = x1[face[leaf][1]];
        Vec3f y12 = x1[face[leaf][2]];
        Vec3f z10 = extrapolate * (y10 - y00) + y00;
        Vec3f z11 = extrapolate * (y11 - y01) + y01;
        Vec3f z12 = extrapolate * (y12 - y02) + y02;
        aabb[index] = aabb::join(aabb::make(y00, y01, y02, ext_eps),
                                 aabb::make(z10, z11, z12, ext_eps));
    } else {
        unsigned left = node[0] - 1;
        unsigned right = node[1] - 1;
        aabb[index] = aabb::join(aabb[left], aabb[right]);
    }
}

__device__ void update_edge_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                                 const BVH &bvh, Vec<AABB> aabb,
                                 const Vec<Vec2u> &edge,
                                 const Vec<EdgeProp> prop,
                                 const Vec<EdgeParam> &edge_params,
                                 unsigned level, const ParamSet &param,
                                 float extrapolate, unsigned i) {
    unsigned index = bvh.level(level, i);
    Vec2u node = bvh.node[index];
    if (node[1] == 0) {
        unsigned leaf = node[0] - 1;
        unsigned i0 = edge[leaf][0];
        unsigned i1 = edge[leaf][1];
        Vec3f y00 = x0[i0];
        Vec3f y01 = x0[i1];
        Vec3f y10 = x1[i0];
        Vec3f y11 = x1[i1];
        Vec3f z10 = extrapolate * (y10 - y00) + y00;
        Vec3f z11 = extrapolate * (y11 - y01) + y01;
        const EdgeParam &eparam = edge_params[prop[leaf].param_index];
        float ext_eps = 0.5f * eparam.ghat + eparam.offset;
        aabb[index] = aabb::join(aabb::make(y00, y01, ext_eps),
                                 aabb::make(z10, z11, ext_eps));
    } else {
        unsigned left = node[0] - 1;
        unsigned right = node[1] - 1;
        aabb[index] = aabb::join(aabb[left], aabb[right]);
    }
}

__device__ void update_vertex_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                                   const BVH &bvh, Vec<AABB> aabb,
                                   const Vec<VertexProp> &prop,
                                   const Vec<VertexParam> &vertex_params,
                                   unsigned level, const ParamSet &param,
                                   float extrapolate, unsigned i) {
    unsigned index = bvh.level(level, i);
    Vec2u node = bvh.node[index];
    if (node[1] == 0) {
        unsigned leaf = node[0] - 1;
        const VertexParam &vparam = vertex_params[prop[leaf].param_index];
        float ext_eps = 0.5f * vparam.ghat + vparam.offset;
        Vec3f y0 = x0[leaf];
        Vec3f y1 = x1[leaf];
        Vec3f z1 = extrapolate * (y1 - y0) + y0;
        aabb[index] = aabb::make(y0, z1, ext_eps);
    } else {
        unsigned left = node[0] - 1;
        unsigned right = node[1] - 1;
        aabb[index] = aabb::join(aabb[left], aabb[right]);
    }
}

void update_aabb(const DataSet &host_data, const DataSet &dev_data,
                 const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                 const ParamSet &param) {

    const MeshInfo &mesh = dev_data.mesh;
    const BVHSet &bvhset = dev_data.bvh;
    float extrapolate = param.line_search_max_t;

    const BVH &face_bvh = bvhset.face;
    Vec<AABB> face_aabb = storage::face_aabb;
    Vec<FaceProp> face_prop = dev_data.prop.face;
    Vec<FaceParam> face_params = dev_data.param_arrays.face;
    Vec<EdgeParam> edge_params = dev_data.param_arrays.edge;
    Vec<VertexParam> vertex_params = dev_data.param_arrays.vertex;

    for (unsigned level = 0; level < bvhset.face.level.size; ++level) {
        unsigned count = host_data.bvh.face.level.count(level);
        DISPATCH_START(count)
        [mesh, face_prop, face_params, param, x0, x1, face_bvh, level,
         extrapolate, face_aabb] __device__(unsigned i) mutable {
            update_face_aabb(x0, x1, face_bvh, face_aabb, mesh.mesh.face,
                             face_prop, face_params, level, param, extrapolate,
                             i);
        } DISPATCH_END;
    }
    const BVH &edge_bvh = bvhset.edge;
    Vec<AABB> edge_aabb = storage::edge_aabb;
    Vec<EdgeProp> edge_prop = dev_data.prop.edge;
    for (unsigned level = 0; level < bvhset.edge.level.size; ++level) {
        unsigned count = host_data.bvh.edge.level.count(level);
        DISPATCH_START(count)
        [mesh, edge_prop, edge_params, param, x0, x1, edge_bvh, level,
         extrapolate, edge_aabb] __device__(unsigned i) mutable {
            update_edge_aabb(x0, x1, edge_bvh, edge_aabb, mesh.mesh.edge,
                             edge_prop, edge_params, level, param, extrapolate,
                             i);
        } DISPATCH_END;
    }
    const BVH &vertex_bvh = bvhset.vertex;
    Vec<AABB> vertex_aabb = storage::vertex_aabb;
    Vec<VertexProp> vertex_prop = dev_data.prop.vertex;
    for (unsigned level = 0; level < bvhset.vertex.level.size; ++level) {
        unsigned count = host_data.bvh.vertex.level.count(level);
        DISPATCH_START(count)
        [mesh, vertex_prop, vertex_params, param, x0, x1, vertex_bvh, level,
         extrapolate, vertex_aabb] __device__(unsigned i) mutable {
            update_vertex_aabb(x0, x1, vertex_bvh, vertex_aabb, vertex_prop,
                               vertex_params, level, param, extrapolate, i);
        } DISPATCH_END;
    }
}

void update_collision_mesh_aabb(const DataSet &host_data,
                                const DataSet &dev_data,
                                const ParamSet &param) {
    const Vec<Vec3f> &vertex = dev_data.constraint.mesh.vertex;
    const BVH &face_bvh = dev_data.constraint.mesh.face_bvh;
    Vec<AABB> face_aabb = storage::collision_mesh_face_aabb;
    Vec<FaceProp> face_prop = dev_data.constraint.mesh.prop.face;
    Vec<EdgeProp> edge_prop = dev_data.constraint.mesh.prop.edge;
    Vec<FaceParam> face_params = dev_data.constraint.mesh.param_arrays.face;
    Vec<EdgeParam> edge_params = dev_data.constraint.mesh.param_arrays.edge;
    const Vec<Vec3u> &face = dev_data.constraint.mesh.face;
    for (unsigned level = 0; level < face_bvh.level.size; ++level) {
        unsigned count = host_data.constraint.mesh.face_bvh.level.count(level);
        DISPATCH_START(count)
        [vertex, face_bvh, face_prop, face_params, param, level, face_aabb,
         face] __device__(unsigned i) mutable {
            update_face_aabb(vertex, vertex, face_bvh, face_aabb, face,
                             face_prop, face_params, level, param, 0.0f, i);
        } DISPATCH_END;
    }
    const BVH &edge_bvh = dev_data.constraint.mesh.edge_bvh;
    Vec<AABB> edge_aabb = storage::collision_mesh_edge_aabb;
    const Vec<Vec2u> &edge = dev_data.constraint.mesh.edge;
    for (unsigned level = 0; level < edge_bvh.level.size; ++level) {
        unsigned count = host_data.constraint.mesh.edge_bvh.level.count(level);
        DISPATCH_START(count)
        [vertex, edge_bvh, edge_prop, edge_params, param, level, edge_aabb,
         edge] __device__(unsigned i) mutable {
            update_edge_aabb(vertex, vertex, edge_bvh, edge_aabb, edge,
                             edge_prop, edge_params, level, param, 0.0f, i);
        } DISPATCH_END;
    }
}

__device__ bool edge_has_shared_vert(const Vec2u &e0, const Vec2u &e1) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (e0[i] == e1[j])
                return true;
        }
    }
    return false;
}

void check_success(Vec<char> array) {
    DISPATCH_START(array.size)[array] __device__(unsigned i) {
        assert(array[i] == 0);
    }
    DISPATCH_END;
}

template <typename F> struct AABB_AABB_Tester {
    __device__ AABB_AABB_Tester(F &op) : op(op) {}
    __device__ bool operator()(unsigned index) { return op(index); }
    __device__ bool test(const AABB &a, const AABB &b) {
        return aabb::overlap(a, b);
    }
    F &op;
};

template <unsigned N>
__device__ void
extend_contact_force_hess(const Proximity<N> &prox, const Vec3f &force,
                          const Mat3x3f &hess, SMatf<3, N> &ext_force,
                          SMatf<N * 3, N * 3> &ext_hess) {
    for (unsigned i = 0; i < N; ++i) {
        ext_force.col(i) = prox.value[i] * force;
    }
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            ext_hess.template block<3, 3>(3 * i, 3 * j) =
                prox.value[i] * prox.value[j] * hess;
        }
    }
}

template <unsigned N>
__device__ static void
atomic_embed_hessian(const Eigen::Vector<unsigned, N> &index,
                     const Eigen::Matrix<float, N * 3, N * 3> &H,
                     FixedCSRMat &fixed, DynCSRMat &dyn) {
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            unsigned i = index[ii];
            unsigned j = index[jj];
            if (i <= j) {
                Mat3x3f val = H.template block<3, 3>(ii * 3, jj * 3);
                if (!fixed.push(i, j, val)) {
                    dyn.push(i, j, val);
                }
            }
        }
    }
}

template <unsigned N>
__device__ static void
dry_atomic_embed_hessian(const Eigen::Vector<unsigned, N> &index,
                         const FixedCSRMat &fixed, DynCSRMat &dyn) {
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            unsigned i = index[ii];
            unsigned j = index[jj];
            if (i <= j) {
                if (!fixed.exists(i, j)) {
                    dyn.dry_push(i, j);
                }
            }
        }
    }
}

template <unsigned N>
__device__ void embed_contact_force_hess(
    const Proximity<N> &prox, const Vec<Vec3f> &x0, const Vec<Vec3f> &x,
    const FixedCSRMat &fixed_in, FixedCSRMat &fixed_out, Vec<float> &force_out,
    const Vec<float> &force_in, DynCSRMat &dyn_out, float ghat, float offset,
    const Vec<VertexProp> &vert_prop, Barrier barrier, float dt, float friction,
    unsigned count, const ParamSet &param, int stage, bool include_friction,
    unsigned i) {

    SVecf<N> mass;
    Vec3f ex0 = Vec3f::Zero();
    Vec3f ex = Vec3f::Zero();
    float area = 0.0f;
    float wsum = 0.0f;
    for (int ii = 0; ii < N; ++ii) {
        unsigned index = prox.index[ii];
        mass[ii] = vert_prop[index].mass;
        float w = prox.value[ii];
        area += fabsf(w) * vert_prop[index].area;
        wsum += fabsf(w);
        ex0 += w * x0[index];
        ex += w * x[index];
    }

    assert(ex0.squaredNorm() > sqr(offset));
    assert(ex.squaredNorm() > sqr(offset));

    Vec3f dx = ex - ex0;
    if (wsum) {
        area /= wsum;
    }
    Vec3f normal = ex.normalized();
    float stiff_k = barrier::compute_stiffness<N>(prox, mass, fixed_in, ex,
                                                  ghat, offset, param);

    if (stage == 0) {
        dry_atomic_embed_hessian<N>(prox.index, fixed_out, dyn_out);
    } else {
        Vec3f f =
            stiff_k * barrier::compute_edge_gradient(ex, ghat, offset, barrier);
        Mat3x3f H =
            stiff_k * barrier::compute_edge_hessian(ex, ghat, offset, barrier);

        if (include_friction) {
            Friction _friction(f, dx, normal, friction, param.friction_eps);
            f += _friction.gradient();
            H += _friction.hessian();
        }

        SMatf<3, N> ext_force;
        SMatf<N * 3, N * 3> ext_hess;
        extend_contact_force_hess<N>(prox, f, H, ext_force, ext_hess);
        utility::atomic_embed_force<N>(prox.index, count * ext_force,
                                       force_out);
        atomic_embed_hessian<N>(prox.index, count * ext_hess, fixed_out,
                                dyn_out);
    }
}

struct PointPointContactForceHessEmbed {

    unsigned vertex_index;
    unsigned rod_count;
    const Vec<Vec2u> &edge;
    const Vec<Vec3u> &face;
    const VertexNeighbor &neighbor;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &prop;
    const Vec<VertexParam> &vertex_params;
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        bool either_dyn =
            prop[vertex_index].fix_index == 0 || prop[index].fix_index == 0;
        if (index < vertex_index && either_dyn) {
            const Vec3f &x = eval_x[vertex_index];
            const Vec3f &y = eval_x[index];
            Vec3f e = x - y;
            const VertexParam &vparam_vertex =
                vertex_params[prop[vertex_index].param_index];
            const VertexParam &vparam_index =
                vertex_params[prop[index].param_index];
            float offset = vparam_vertex.offset + vparam_index.offset;
            float friction =
                std::max(vparam_vertex.friction, vparam_index.friction);
            float ghat = 0.5f * (vparam_vertex.ghat + vparam_index.ghat);
            if (e.squaredNorm() < sqr(ghat + offset)) {
                unsigned count = 0;
                bool include_friction = true;
                if (neighbor.edge.count(index)) {
                    for (unsigned j = neighbor.edge.offset[index];
                         j < neighbor.edge.offset[index + 1]; ++j) {
                        Vec2u f = edge[neighbor.edge.data[j]];
                        if (f[0] == vertex_index || f[1] == vertex_index) {
                            ++count;
                            include_friction = false;
                        } else {
                            Vec2f c = distance::point_edge_distance_coeff<
                                float, float>(x, eval_x[f[0]], eval_x[f[1]]);
                            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                                continue;
                            } else {
                                ++count;
                            }
                        }
                    }
                    if (neighbor.face.count(index)) {
                        for (unsigned j = neighbor.face.offset[index];
                             j < neighbor.face.offset[index + 1]; ++j) {
                            Vec3u f = face[neighbor.face.data[j]];
                            if (f[0] == vertex_index || f[1] == vertex_index ||
                                f[2] == vertex_index) {
                                ++count;
                                include_friction = false;
                            } else {
                                Vec3f c =
                                    distance::point_triangle_distance_coeff<
                                        float, float>(x, eval_x[f[0]],
                                                      eval_x[f[1]],
                                                      eval_x[f[2]]);
                                if (c.maxCoeff() < 1.0f &&
                                    c.minCoeff() > 0.0f) {
                                    continue;
                                } else {
                                    ++count;
                                }
                            }
                        }
                    }
                } else {
                    count = 1u;
                }
                if (count) {
                    assert(e.squaredNorm() > sqr(offset));
                    Proximity<2> prox;
                    prox.index = Vec2u(vertex_index, index);
                    prox.value = Vec2f(1.0f, -1.0f);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, ghat, offset, prop, param.barrier,
                        dt, friction, count, param, stage, include_friction,
                        vertex_index);
                    return true;
                }
            }
        }
        return false;
    }
};

struct PointEdgeContactForceHessEmbed {

    unsigned vertex_index;
    const Vec<Vec2u> &edge;
    const Vec<Vec3u> &face;
    const EdgeNeighbor &neighbor;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &vert_prop;
    const Vec<EdgeProp> &edge_prop;
    const Vec<VertexParam> &vertex_params;
    const Vec<EdgeParam> &edge_params;
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        Vec2u f = edge[index];
        bool either_dyn = vert_prop[vertex_index].fix_index == 0 ||
                          edge_prop[index].fixed == false;
        if (f[0] != vertex_index && f[1] != vertex_index && either_dyn) {
            const Vec3f &p = eval_x[vertex_index];
            const Vec3f &t0 = eval_x[f[0]];
            const Vec3f &t1 = eval_x[f[1]];
            Vec2f c =
                distance::point_edge_distance_coeff<float, float>(p, t0, t1);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f e = c[0] * (p - t0) + c[1] * (p - t1);
                const VertexParam &vparam =
                    vertex_params[vert_prop[vertex_index].param_index];
                const EdgeParam &eparam =
                    edge_params[edge_prop[index].param_index];
                float offset = vparam.offset + eparam.offset;
                float ghat = 0.5f * (vparam.ghat + eparam.ghat);
                float friction = std::max(vparam.friction, eparam.friction);
                if (e.squaredNorm() < sqr(ghat + offset)) {
                    unsigned count = 0;
                    bool include_friction = true;
                    if (neighbor.face.count(index)) {
                        for (unsigned j = neighbor.face.offset[index];
                             j < neighbor.face.offset[index + 1]; ++j) {
                            Vec3u f = face[neighbor.face.data[j]];
                            if (f[0] == vertex_index || f[1] == vertex_index ||
                                f[2] == vertex_index) {
                                ++count;
                                include_friction = false;
                            } else {
                                Vec3f c =
                                    distance::point_triangle_distance_coeff<
                                        float, float>(p, eval_x[f[0]],
                                                      eval_x[f[1]],
                                                      eval_x[f[2]]);
                                if (c.maxCoeff() < 1.0f &&
                                    c.minCoeff() > 0.0f) {
                                    continue;
                                } else {
                                    ++count;
                                }
                            }
                        }
                    } else {
                        count = 1u;
                    }
                    if (count) {
                        assert(e.squaredNorm() > sqr(offset));
                        Proximity<3> prox;
                        prox.index = Vec3u(vertex_index, f[0], f[1]);
                        prox.value = Vec3f(1.0f, -c[0], -c[1]);
                        embed_contact_force_hess(
                            prox, vertex, eval_x, fixed_hess_in, fixed_out,
                            force, force_in, dyn_out, ghat, offset, vert_prop,
                            param.barrier, dt, friction, count, param, stage,
                            include_friction, vertex_index);
                        return true;
                    }
                }
            }
        }
        return false;
    }
};

struct PointFaceContactForceHessEmbed {

    unsigned vertex_index;
    const Vec<Vec3u> &face;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &vert_prop;
    const Vec<FaceProp> &face_prop;
    const Vec<VertexParam> &vertex_params;
    const Vec<FaceParam> &face_params;
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        Vec3u f = face[index];
        bool either_dyn = vert_prop[vertex_index].fix_index == 0 ||
                          face_prop[index].fixed == false;
        if (f[0] != vertex_index && f[1] != vertex_index &&
            f[2] != vertex_index && either_dyn) {
            const Vec3f &p = eval_x[vertex_index];
            const Vec3f &t0 = eval_x[f[0]];
            const Vec3f &t1 = eval_x[f[1]];
            const Vec3f &t2 = eval_x[f[2]];
            Vec3f c = distance::point_triangle_distance_coeff<float, float>(
                p, t0, t1, t2);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f e = c[0] * (p - t0) + c[1] * (p - t1) + c[2] * (p - t2);
                const VertexParam &vparam =
                    vertex_params[vert_prop[vertex_index].param_index];
                const FaceParam &fparam =
                    face_params[face_prop[index].param_index];
                float offset = vparam.offset + fparam.offset;
                float ghat = 0.5f * (vparam.ghat + fparam.ghat);
                float friction = std::max(vparam.friction, fparam.friction);
                if (e.squaredNorm() < sqr(ghat + offset)) {
                    assert(e.squaredNorm() > sqr(offset));
                    Proximity<4> prox;
                    prox.index = Vec4u(vertex_index, f[0], f[1], f[2]);
                    prox.value = Vec4f(1.0f, -c[0], -c[1], -c[2]);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, ghat, offset, vert_prop,
                        param.barrier, dt, friction, 1, param, stage, true,
                        vertex_index);
                    return true;
                }
            }
        }
        return false;
    }
};

struct EdgeEdgeContactForceHessEmbed {

    unsigned edge_index;
    const Vec<Vec2u> &edge;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &vert_prop;
    const Vec<EdgeProp> &edge_prop;
    const Vec<EdgeParam> &edge_params;
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        const Vec2u &e0 = edge[edge_index];
        const Vec2u &e1 = edge[index];
        bool either_dyn = edge_prop[edge_index].fixed == false ||
                          edge_prop[index].fixed == false;
        if (edge_index < index && edge_has_shared_vert(e0, e1) == false &&
            either_dyn) {
            const Vec3f &p0 = eval_x[e0[0]];
            const Vec3f &p1 = eval_x[e0[1]];
            const Vec3f &q0 = eval_x[e1[0]];
            const Vec3f &q1 = eval_x[e1[1]];
            Vec4f c = distance::edge_edge_distance_coeff<float, float>(p0, p1,
                                                                       q0, q1);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f x0 = c[0] * p0 + c[1] * p1;
                Vec3f x1 = c[2] * q0 + c[3] * q1;
                Vec3f e = x0 - x1;
                const EdgeParam &eparam_edge =
                    edge_params[edge_prop[edge_index].param_index];
                const EdgeParam &eparam_index =
                    edge_params[edge_prop[index].param_index];
                float offset = eparam_edge.offset + eparam_index.offset;
                float ghat = 0.5f * (eparam_edge.ghat + eparam_index.ghat);
                float friction =
                    std::max(eparam_edge.friction, eparam_index.friction);
                if (e.squaredNorm() < sqr(ghat + offset)) {
                    assert(e.squaredNorm() > sqr(offset));
                    Proximity<4> prox;
                    prox.index = Vec4u(e0[0], e0[1], e1[0], e1[1]);
                    prox.value = Vec4f(c[0], c[1], -c[2], -c[3]);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, ghat, offset, vert_prop,
                        param.barrier, dt, friction, 1, param, stage, true,
                        edge_index);
                    return true;
                }
            }
        }
        return false;
    }
};

struct CollisionHessForceEmbedArgs {
    const Vec<Vec3f> collision_mesh_vertex;
    const Vec<Vec3u> collision_mesh_face;
    const Vec<Vec2u> collision_mesh_edge;
    const BVH collision_mesh_face_bvh;
    const BVH collision_mesh_edge_bvh;
    const Vec<AABB> collision_mesh_face_aabb;
    const Vec<AABB> collision_mesh_edge_aabb;
};

struct CollisionMeshVertexFaceContactForceHessEmbed_M2C {

    unsigned vertex_index;
    Vec<float> force;
    const Mat3x3f &local_hess;
    FixedCSRMat &dyn_out;
    const Vec<Vec3u> &collision_mesh_face;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<FaceProp> &static_face_prop;
    const Vec<VertexParam> &dyn_vert_params;
    const Vec<FaceParam> &static_face_params;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (dyn_vert_prop[vertex_index].fix_index == 0) {
            const Vec3u &f = collision_mesh_face[index];
            Vec3f p = eval_x[vertex_index];
            Vec3f q = vertex[vertex_index] - p;
            Vec3f t0 = collision_mesh_vertex[f[0]] - p;
            Vec3f t1 = collision_mesh_vertex[f[1]] - p;
            Vec3f t2 = collision_mesh_vertex[f[2]] - p;
            Vec3f zero = Vec3f::Zero();
            Vec3f c = distance::point_triangle_distance_coeff_unclassified<
                float, float>(zero, t0, t1, t2);
            Vec3f y = c[0] * t0 + c[1] * t1 + c[2] * t2;
            Vec3f e = -y;
            const VertexParam &dyn_vparam =
                dyn_vert_params[dyn_vert_prop[vertex_index].param_index];
            const FaceParam &static_fparam =
                static_face_params[static_face_prop[index].param_index];
            float offset = dyn_vparam.offset + static_fparam.offset;
            float ghat = 0.5f * (dyn_vparam.ghat + static_fparam.ghat);
            float friction =
                std::max(dyn_vparam.friction, static_fparam.friction);
            if (e.squaredNorm() < sqr(offset + ghat)) {
                assert(e.squaredNorm() > sqr(offset));
                Vec3f normal = e.normalized();
                float mass = dyn_vert_prop[vertex_index].mass;
                Vec3f proj_x = y + normal * (offset + ghat);
                float gap_squared = sqr(e.norm() - offset);
                float stiff_k =
                    normal.dot(local_hess * normal) + mass / gap_squared;
                Vec3f f = stiff_k * push::gradient(-proj_x, normal, ghat);
                Mat3x3f H = stiff_k * push::hessian(-proj_x, normal, ghat);
                Friction _friction(f, -q, normal, friction, param.friction_eps);
                f += _friction.gradient();
                H += _friction.hessian();

                utility::atomic_embed_force<1>(Vec1u(vertex_index), f, force);
                utility::atomic_embed_hessian<1>(Vec1u(vertex_index), H,
                                                 dyn_out);
                return true;
            }
        }
        return false;
    }
};

struct CollisionMeshVertexFaceContactForceHessEmbed_C2M {

    const DataSet &data;
    unsigned vertex_index;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    Vec<float> force;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<FaceProp> &dyn_face_prop;
    const Vec<VertexProp> &static_vert_prop;
    const Vec<FaceParam> &dyn_face_params;
    const Vec<VertexParam> &static_vert_params;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (!dyn_face_prop[index].fixed) {
            const Vec3u &fc = data.mesh.mesh.face[index];
            const Vec3f &y = data.constraint.mesh.vertex[vertex_index];
            const Vec3f t0 = eval_x[fc[0]] - y;
            const Vec3f t1 = eval_x[fc[1]] - y;
            const Vec3f t2 = eval_x[fc[2]] - y;
            const Vec3f s0 = vertex[fc[0]] - y;
            const Vec3f s1 = vertex[fc[1]] - y;
            const Vec3f s2 = vertex[fc[2]] - y;
            const Vec3f zero = Vec3f::Zero();
            const Vec3f c =
                distance::point_triangle_distance_coeff_unclassified<float,
                                                                     float>(
                    zero, t0, t1, t2);
            Vec3f p = c[0] * t0 + c[1] * t1 + c[2] * t2;
            Vec3f q = c[0] * s0 + c[1] * s1 + c[2] * s2;
            const Vec3f &e = p;
            Vec3f dx = p - q;
            const FaceParam &dyn_fparam =
                dyn_face_params[dyn_face_prop[index].param_index];
            const VertexParam &static_vparam =
                static_vert_params[static_vert_prop[vertex_index].param_index];
            float offset = dyn_fparam.offset + static_vparam.offset;
            float ghat = 0.5f * (dyn_fparam.ghat + static_vparam.ghat);
            float friction =
                std::max(dyn_fparam.friction, static_vparam.friction);
            if (e.squaredNorm() < sqr(offset + ghat)) {
                assert(e.squaredNorm() > sqr(offset));
                Vec3f normal = e.normalized();
                Mat9x9f local_hess = Mat9x9f::Zero();
                float gap_squared = sqr(e.norm() - offset);
                for (unsigned ii = 0; ii < 3; ++ii) {
                    for (unsigned jj = 0; jj < 3; ++jj) {
                        local_hess.block<3, 3>(3 * ii, 3 * jj) =
                            fixed_hess_in(fc[ii], fc[jj]);
                    }
                    local_hess.block<3, 3>(3 * ii, 3 * ii) +=
                        (dyn_vert_prop[fc[ii]].mass / gap_squared) *
                        Mat3x3f::Identity();
                }
                Vec9f normal_ext;
                for (int j = 0; j < 9; ++j) {
                    normal_ext[j] = c[j / 3] * normal[j % 3];
                }
                float stiff_k = (local_hess * normal_ext).dot(normal_ext) /
                                normal_ext.squaredNorm();
                float area = c[0] * dyn_vert_prop[fc[0]].area +
                             c[1] * dyn_vert_prop[fc[1]].area +
                             c[2] * dyn_vert_prop[fc[2]].area;

                Vec3f proj_x = normal * (offset + ghat);
                Vec3f f = stiff_k * push::gradient(p - proj_x, normal, ghat);
                Mat3x3f H = stiff_k * push::hessian(p - proj_x, normal, ghat);
                Friction _friction(f, p - q, normal, friction,
                                   param.friction_eps);
                f += _friction.gradient();
                H += _friction.hessian();
                Mat3x3f ff;
                Mat9x9f HH;
                for (int i = 0; i < 3; ++i) {
                    ff.col(i) = c[i] * f;
                }
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        HH.block<3, 3>(i * 3, j * 3) = c[i] * c[j] * H;
                    }
                }
                utility::atomic_embed_force<3>(fc, ff, force);
                utility::atomic_embed_hessian<3>(fc, HH, fixed_out);
                return true;
            }
        }
        return false;
    }
};

struct CollisionMeshEdgeEdgeContactForceHessEmbed {

    unsigned i;
    const Vec<Vec2u> &mesh_edge;
    Vec<float> &force;
    const Mat6x6f &local_hess;
    FixedCSRMat &dyn_out;
    const Vec<Vec2u> &collision_mesh_edge;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<EdgeProp> &dyn_edge_prop;
    const Vec<EdgeProp> &static_edge_prop;
    const Vec<EdgeParam> &dyn_edge_params;
    const Vec<EdgeParam> &static_edge_params;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (!dyn_edge_prop[i].fixed) {
            const Vec2u &mesh_edge = this->mesh_edge[i];
            const Vec2u &coll_edge = collision_mesh_edge[index];
            Vec3f p0 = eval_x[mesh_edge[0]];
            Vec3f p1 = eval_x[mesh_edge[1]];
            Vec3f q0 = collision_mesh_vertex[coll_edge[0]];
            Vec3f q1 = collision_mesh_vertex[coll_edge[1]];
            Vec3f b0 = vertex[mesh_edge[0]];
            Vec3f b1 = vertex[mesh_edge[1]];
            Vec4f c =
                distance::edge_edge_distance_coeff_unclassified<float, float>(
                    p0, p1, q0, q1);
            Vec3f x = c[0] * p0 + c[1] * p1;
            Vec3f y = c[2] * q0 + c[3] * q1;
            Vec3f z = c[0] * b0 + c[1] * b1;
            Vec3f e = x - y;
            Vec3f dx = x - z;
            const EdgeParam &dyn_eparam =
                dyn_edge_params[dyn_edge_prop[i].param_index];
            const EdgeParam &static_eparam =
                static_edge_params[static_edge_prop[index].param_index];
            float offset = dyn_eparam.offset + static_eparam.offset;
            float ghat = 0.5f * (dyn_eparam.ghat + static_eparam.ghat);
            float friction =
                std::max(dyn_eparam.friction, static_eparam.friction);
            if (e.squaredNorm() < sqr(offset + ghat)) {
                assert(e.squaredNorm() > sqr(offset));
                Vec3f normal = e.normalized();
                Vec3f proj_x = (offset + ghat) * normal;
                Vec6f normal_ext;
                for (int j = 0; j < 6; ++j) {
                    normal_ext[j] = c[j / 3] * normal[j % 3];
                }
                Mat6x6f mass_diag = Mat6x6f::Zero();
                float gap_squared = sqr(e.norm() - offset);
                mass_diag.block<3, 3>(0, 0) =
                    (dyn_vert_prop[mesh_edge[0]].mass / gap_squared) *
                    Mat3x3f::Identity();
                mass_diag.block<3, 3>(3, 3) =
                    (dyn_vert_prop[mesh_edge[1]].mass / gap_squared) *
                    Mat3x3f::Identity();
                float stiff_k =
                    ((local_hess + mass_diag) * normal_ext).dot(normal_ext) /
                    normal_ext.squaredNorm();
                float area = c[0] * dyn_vert_prop[mesh_edge[0]].area +
                             c[1] * dyn_vert_prop[mesh_edge[1]].area;
                Vec3f f = stiff_k * push::gradient(e - proj_x, normal, ghat);
                Mat3x3f H = stiff_k * push::hessian(e - proj_x, normal, ghat);

                Friction _friction(f, x - z, normal, friction,
                                   param.friction_eps);
                f += _friction.gradient();
                H += _friction.hessian();

                Mat3x2f ff;
                Mat6x6f HH;
                for (int i = 0; i < 2; ++i) {
                    ff.col(i) = c[i] * f;
                }
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        HH.block<3, 3>(i * 3, j * 3) = c[i] * c[j] * H;
                    }
                }
                utility::atomic_embed_force<2>(mesh_edge, ff, force);
                utility::atomic_embed_hessian<2>(mesh_edge, HH, dyn_out);
                return true;
            }
        }
        return false;
    }
};

__device__ unsigned embed_vertex_constraint_force_hessian(
    const DataSet &data, const Vec<Vec3f> &eval_x, Vec<float> &force,
    const FixedCSRMat &fixed_hess_in, FixedCSRMat &fixed_out, float dt,
    const ParamSet &param, unsigned i) {

    const VertexProp &prop = data.prop.vertex[i];
    const VertexParam &vparam = data.param_arrays.vertex[prop.param_index];
    float mass = prop.mass;
    unsigned num_contact = 0;

    Mat3x3f local_hess = fixed_hess_in(i, i);
    Mat3x3f H = Mat3x3f::Zero();
    Vec3f f = Vec3f::Zero();
    const Vec3f &x = eval_x[i];
    const Vec3f &dx = x - data.vertex.curr[i];

    if (prop.fix_index > 0) {
        const FixPair &fix = data.constraint.fix[prop.fix_index - 1];
        const Vec3f &y = fix.position;
        Vec3f w = x - y;
        float d = w.norm();
        float gap = fix.ghat - d;
        if (fix.kinematic) {
            gap = fmaxf(gap, param.constraint_tol * fix.ghat);
        } else {
            assert(gap >= 0.0f);
        }
        float tmp =
            w.squaredNorm() ? (local_hess * w).dot(w) / w.squaredNorm() : 0.0f;
        float stiff_k = tmp + mass / (gap * gap);
        f += stiff_k * fix::gradient(x, y);
        H += stiff_k * fix::hessian();
    } else {
        for (unsigned j = 0; j < data.constraint.sphere.size; ++j) {
            const Sphere &sphere = data.constraint.sphere[j];
            float ghat = sphere.ghat;
            float friction = std::max(sphere.friction, vparam.friction);
            bool bowl = sphere.bowl;
            bool reverse = sphere.reverse;
            float radius = sphere.radius;
            Vec3f center = sphere.center;
            center = (bowl && (x[1] > center[1]))
                         ? Vec3f(center[0], x[1], center[2])
                         : center;
            float d2 = (x - center).squaredNorm();
            if (sphere.kinematic) {
                if (d2) {
                    Vec3f normal = (x - center).normalized();
                    Vec3f target = radius * normal;
                    Vec3f o = x - center;
                    if (bowl) {
                        reverse = true;
                    }
                    float stiff_k = mass / (ghat * ghat);
                    float r2 = radius * radius;
                    if (reverse == true && d2 > r2) {
                        f +=
                            stiff_k * push::gradient(o - target, -normal, ghat);
                        H += stiff_k * push::hessian(o - target, -normal, ghat);
                    } else if (reverse == false && d2 < r2) {
                        f += stiff_k * push::gradient(o - target, normal, ghat);
                        H += stiff_k * push::hessian(o - target, normal, ghat);
                    }
                }
            } else {
                if (reverse) {
                    radius -= ghat;
                } else {
                    radius += ghat;
                }
                float r2 = radius * radius;
                bool intersected = reverse ? d2 > r2 : d2 < r2;
                if (intersected) {
                    num_contact += 1;
                    Vec3f normal = (x - center).normalized();
                    Vec3f projected_x = radius * normal;
                    Vec3f o = x - center;
                    if (reverse) {
                        normal = -normal;
                    }
                    float gap;
                    if (reverse) {
                        gap = sphere.radius - sqrtf(d2);
                    } else {
                        gap = sqrtf(d2) - sphere.radius;
                    }
                    assert(gap >= 0.0f);
                    float stiff_k =
                        (normal.dot(local_hess * normal) + mass / (gap * gap));
                    Vec3f f_push =
                        stiff_k * push::gradient(o - projected_x, normal, ghat);
                    f += f_push;
                    H += stiff_k * push::hessian(o - projected_x, normal, ghat);

                    Friction _friction(f_push, dx, normal, friction,
                                       param.friction_eps);
                    f += _friction.gradient();
                    H += _friction.hessian();
                }
            }
        }

        for (unsigned j = 0; j < data.constraint.floor.size; ++j) {
            const Floor &floor = data.constraint.floor[j];
            float ghat = floor.ghat;
            float friction = std::max(floor.friction, vparam.friction);
            const Vec3f &up = floor.up;
            Vec3f ground = floor.ground + (floor.kinematic ? 0.0f : ghat) * up;

            Vec3f e = x - ground;
            if (e.dot(up) < 0.0f) {
                num_contact += 1;
                Vec3f projected_x = -e.dot(up) * up;
                float gap = (x - floor.ground).dot(up);
                if (floor.kinematic) {
                    gap = fmaxf(gap, param.constraint_tol * ghat);
                }
                assert(gap >= 0.0f);
                float stiff_k = (up.dot(local_hess * up) + mass / (gap * gap));
                Vec3f f_push = stiff_k * push::gradient(-projected_x, up, ghat);
                Mat3x3f H_push =
                    stiff_k * push::hessian(-projected_x, up, ghat);
                f += f_push;
                H += H_push;
                Friction _friction(f_push, dx, up, friction,
                                   param.friction_eps);
                f += _friction.gradient();
                H += _friction.hessian();
            }
        }
    }

    utility::atomic_embed_force<1>(Vec1u(i), f, force);
    utility::atomic_embed_hessian<1>(Vec1u(i), H, fixed_out);

    return num_contact;
}

unsigned embed_contact_force_hessian(const DataSet &data,
                                     const Vec<Vec3f> &eval_x, Vec<float> force,
                                     const FixedCSRMat &fixed_hess_in,
                                     FixedCSRMat &fixed_out, DynCSRMat &dyn_out,
                                     unsigned &max_nnz_row, float &dyn_consumed,
                                     float dt, const ParamSet &param) {

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned rod_count = data.rod_count;
    unsigned edge_count = data.mesh.mesh.edge.size;
    unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
    unsigned max_contact_vert = (surface_vert_count > collision_mesh_vert_count)
                                    ? surface_vert_count
                                    : collision_mesh_vert_count;
    const BVH face_bvh = data.bvh.face;
    const BVH edge_bvh = data.bvh.edge;
    const BVH vertex_bvh = data.bvh.vertex;
    const Vec<AABB> face_aabb = storage::face_aabb;
    const Vec<AABB> edge_aabb = storage::edge_aabb;
    const Vec<AABB> vertex_aabb = storage::vertex_aabb;

    Vec<VertexParam> vertex_params = data.param_arrays.vertex;
    Vec<EdgeParam> edge_params = data.param_arrays.edge;
    Vec<FaceParam> face_params = data.param_arrays.face;

    buffer::MemoryPool &pool = buffer::get();
    auto contact_force = pool.get<float>(3 * surface_vert_count);
    auto num_contact_vtf = pool.get<unsigned>(max_contact_vert);
    auto num_contact_ee = pool.get<unsigned>(edge_count);

    num_contact_vtf.clear(0);
    num_contact_ee.clear(0);
    contact_force.clear(0.0f);

    Vec<float> contact_force_vec = contact_force.as_vec();
    Vec<unsigned> num_contact_vtf_vec = num_contact_vtf.as_vec();
    Vec<unsigned> num_contact_ee_vec = num_contact_ee.as_vec();

    for (int stage = 0; stage < 2; ++stage) {
        if (stage == 0) {
            dyn_out.start_rebuild_buffer();
        }

        DISPATCH_START(surface_vert_count)
        [data, eval_x, rod_count, contact_force_vec, force, fixed_hess_in,
         fixed_out, dyn_out, face_bvh, face_aabb, edge_bvh, edge_aabb,
         vertex_bvh, vertex_aabb, num_contact_vtf_vec, vertex_params,
         edge_params, face_params, stage, dt,
         param] __device__(unsigned i) mutable {
            unsigned count(0);
            const VertexParam &vparam =
                vertex_params[data.prop.vertex[i].param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            AABB pt_aabb = aabb::make(eval_x[i], ext_eps);

            PointFaceContactForceHessEmbed embed_0 = {i,
                                                      data.mesh.mesh.face,
                                                      data.vertex.curr,
                                                      eval_x,
                                                      contact_force_vec,
                                                      force,
                                                      fixed_hess_in,
                                                      fixed_out,
                                                      dyn_out,
                                                      data.prop.vertex,
                                                      data.prop.face,
                                                      vertex_params,
                                                      face_params,
                                                      stage,
                                                      dt,
                                                      param};

            AABB_AABB_Tester<PointFaceContactForceHessEmbed> op_0(embed_0);
            count += aabb::query(face_bvh, face_aabb, op_0, pt_aabb);

            PointEdgeContactForceHessEmbed embed_1 = {i,
                                                      data.mesh.mesh.edge,
                                                      data.mesh.mesh.face,
                                                      data.mesh.neighbor.edge,
                                                      data.vertex.curr,
                                                      eval_x,
                                                      contact_force_vec,
                                                      force,
                                                      fixed_hess_in,
                                                      fixed_out,
                                                      dyn_out,
                                                      data.prop.vertex,
                                                      data.prop.edge,
                                                      vertex_params,
                                                      edge_params,
                                                      stage,
                                                      dt,
                                                      param};

            AABB_AABB_Tester<PointEdgeContactForceHessEmbed> op_1(embed_1);
            count += aabb::query(edge_bvh, edge_aabb, op_1, pt_aabb);

            PointPointContactForceHessEmbed embed_2 = {
                i,
                rod_count,
                data.mesh.mesh.edge,
                data.mesh.mesh.face,
                data.mesh.neighbor.vertex,
                data.vertex.curr,
                eval_x,
                contact_force_vec,
                force,
                fixed_hess_in,
                fixed_out,
                dyn_out,
                data.prop.vertex,
                vertex_params,
                stage,
                dt,
                param};
            AABB_AABB_Tester<PointPointContactForceHessEmbed> op_2(embed_2);
            count += aabb::query(vertex_bvh, vertex_aabb, op_2, pt_aabb);
            if (stage == 0) {
                num_contact_vtf_vec[i] += count;
            }
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, eval_x, contact_force_vec, force, fixed_hess_in, fixed_out,
         dyn_out, edge_bvh, edge_aabb, num_contact_ee_vec, edge_params, stage,
         dt, param] __device__(unsigned i) mutable {
            Vec2u edge = data.mesh.mesh.edge[i];
            const EdgeParam &eparam =
                edge_params[data.prop.edge[i].param_index];
            float ext_eps = 0.5f * eparam.ghat + eparam.offset;
            AABB aabb = aabb::make(eval_x[edge[0]], eval_x[edge[1]], ext_eps);
            EdgeEdgeContactForceHessEmbed embed = {i,
                                                   data.mesh.mesh.edge,
                                                   data.vertex.curr,
                                                   eval_x,
                                                   contact_force_vec,
                                                   force,
                                                   fixed_hess_in,
                                                   fixed_out,
                                                   dyn_out,
                                                   data.prop.vertex,
                                                   data.prop.edge,
                                                   edge_params,
                                                   stage,
                                                   dt,
                                                   param};
            AABB_AABB_Tester<EdgeEdgeContactForceHessEmbed> op(embed);
            unsigned count = aabb::query(edge_bvh, edge_aabb, op, aabb);
            if (stage == 0) {
                num_contact_ee_vec[i] += count;
            }
        } DISPATCH_END;

        if (stage == 0) {
            // Name: Time for Rebuilding Memory Layout for Contact Matrix
            // Format: list[(vid_time,ms)]
            // Map: contact_mat_rebuild
            // Description:
            // After the dry pass, the memory layout for the contact matrix
            // is re-computed so that the matrix can be assembled in the
            // fill-in pass.
            dyn_out.finish_rebuild_buffer(max_nnz_row, dyn_consumed);
        } else {
            // Name: Time for Filializing Contact Matrix
            // Format: list[(vid_time,ms)]
            // Map: contact_mat_finalize
            // Description:
            // After the fill-in pass, the contact matrix is compressed to
            // eliminate redundant entries.
            if (dyn_consumed) {
                dyn_out.finalize();
            }
        }
    }

    DISPATCH_START(3 * surface_vert_count)
    [force, contact_force_vec] __device__(unsigned i) mutable {
        force[i] += contact_force_vec[i];
    } DISPATCH_END;

    unsigned n_contact =
        kernels::sum_array(num_contact_vtf.data, num_contact_vtf.size) +
        kernels::sum_array(num_contact_ee.data, num_contact_ee.size);

    // PooledVec buffers auto-release when exiting function
    return n_contact;
}

unsigned embed_constraint_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> force,
                                        const FixedCSRMat &fixed_hess_in,
                                        FixedCSRMat &fixed_out, float dt,
                                        const ParamSet &param) {

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned edge_count = data.mesh.mesh.edge.size;
    unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
    unsigned max_contact_vert = (surface_vert_count > collision_mesh_vert_count)
                                    ? surface_vert_count
                                    : collision_mesh_vert_count;

    const BVH &face_bvh = data.bvh.face;
    const Vec<AABB> face_aabb = storage::face_aabb;

    Vec<VertexParam> vertex_params = data.param_arrays.vertex;
    Vec<EdgeParam> edge_params = data.param_arrays.edge;
    Vec<FaceParam> face_params = data.param_arrays.face;
    Vec<VertexParam> collision_vertex_params =
        data.constraint.mesh.param_arrays.vertex;
    Vec<EdgeParam> collision_edge_params =
        data.constraint.mesh.param_arrays.edge;
    Vec<FaceParam> collision_face_params =
        data.constraint.mesh.param_arrays.face;

    buffer::MemoryPool &pool = buffer::get();
    auto num_contact_vtf = pool.get<unsigned>(max_contact_vert);
    auto num_contact_ee = pool.get<unsigned>(edge_count);

    num_contact_vtf.clear(0);
    num_contact_ee.clear(0);

    Vec<unsigned> num_contact_vtf_vec = num_contact_vtf.as_vec();
    Vec<unsigned> num_contact_ee_vec = num_contact_ee.as_vec();

    DISPATCH_START(surface_vert_count)
    [data, eval_x, force, fixed_hess_in, fixed_out, dt, num_contact_vtf_vec,
     param] __device__(unsigned i) mutable {
        num_contact_vtf_vec[i] += embed_vertex_constraint_force_hessian(
            data, eval_x, force, fixed_hess_in, fixed_out, dt, param, i);
    } DISPATCH_END;

    if (!param.disable_contact) {
        CollisionHessForceEmbedArgs args = {
            data.constraint.mesh.vertex,      data.constraint.mesh.face,
            data.constraint.mesh.edge,        data.constraint.mesh.face_bvh,
            data.constraint.mesh.edge_bvh,    storage::collision_mesh_face_aabb,
            storage::collision_mesh_edge_aabb};

        DISPATCH_START(surface_vert_count)
        [data, eval_x, force, fixed_hess_in, fixed_out, args, dt,
         num_contact_vtf_vec, vertex_params, collision_face_params,
         param] __device__(unsigned i) mutable {
            Mat3x3f local_hess = fixed_hess_in(i, i);
            const VertexProp &prop = data.prop.vertex[i];
            const VertexParam &vparam = vertex_params[prop.param_index];
            CollisionMeshVertexFaceContactForceHessEmbed_M2C embed = {
                i,
                force,
                local_hess,
                fixed_out,
                args.collision_mesh_face,
                args.collision_mesh_vertex,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                data.constraint.mesh.prop.face,
                vertex_params,
                collision_face_params,
                dt,
                param};
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            AABB pt_aabb = aabb::make(eval_x[i], ext_eps);
            AABB_AABB_Tester<CollisionMeshVertexFaceContactForceHessEmbed_M2C>
                op(embed);
            num_contact_vtf_vec[i] +=
                aabb::query(args.collision_mesh_face_bvh,
                            args.collision_mesh_face_aabb, op, pt_aabb);
        } DISPATCH_END;

        DISPATCH_START(data.constraint.mesh.vertex.size)
        [data, eval_x, force, fixed_hess_in, fixed_out, args,
         num_contact_vtf_vec, face_bvh, face_aabb, face_params,
         collision_vertex_params, dt, param] __device__(unsigned i) mutable {
            const VertexProp &prop = data.constraint.mesh.prop.vertex[i];
            const VertexParam &vparam =
                collision_vertex_params[prop.param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            CollisionMeshVertexFaceContactForceHessEmbed_C2M embed = {
                data,
                i,
                fixed_hess_in,
                fixed_out,
                force,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                data.prop.face,
                data.constraint.mesh.prop.vertex,
                face_params,
                collision_vertex_params,
                dt,
                param};
            AABB_AABB_Tester<CollisionMeshVertexFaceContactForceHessEmbed_C2M>
                op(embed);
            num_contact_vtf_vec[i] += aabb::query(
                face_bvh, face_aabb, op,
                aabb::make(data.constraint.mesh.vertex[i], ext_eps));
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, eval_x, force, fixed_hess_in, fixed_out, args,
         num_contact_ee_vec, edge_params, collision_edge_params, dt,
         param] __device__(unsigned i) mutable {
            const Vec2u &edge = data.mesh.mesh.edge[i];
            const EdgeParam &eparam =
                edge_params[data.prop.edge[i].param_index];
            float ext_eps = 0.5f * eparam.ghat + eparam.offset;
            Mat6x6f local_hess = Mat6x6f::Zero();
            for (unsigned ii = 0; ii < 2; ++ii) {
                for (unsigned jj = 0; jj < 2; ++jj) {
                    local_hess.block<3, 3>(3 * ii, 3 * jj) =
                        fixed_hess_in(edge[ii], edge[jj]);
                }
            }
            CollisionMeshEdgeEdgeContactForceHessEmbed embed = {
                i,
                data.mesh.mesh.edge,
                force,
                local_hess,
                fixed_out,
                data.constraint.mesh.edge,
                data.constraint.mesh.vertex,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                data.prop.edge,
                data.constraint.mesh.prop.edge,
                edge_params,
                collision_edge_params,
                dt,
                param};
            AABB aabb = aabb::make(eval_x[edge[0]], eval_x[edge[1]], ext_eps);
            AABB_AABB_Tester<CollisionMeshEdgeEdgeContactForceHessEmbed> op(
                embed);
            num_contact_ee_vec[i] +=
                aabb::query(args.collision_mesh_edge_bvh,
                            args.collision_mesh_edge_aabb, op, aabb);
        } DISPATCH_END;
    }

    unsigned n_contact =
        kernels::sum_array(num_contact_vtf_vec.data, num_contact_vtf_vec.size) +
        kernels::sum_array(num_contact_ee_vec.data, num_contact_ee_vec.size);

    return n_contact;
}

struct CollisionMeshPointFaceCCD_M2C {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<Vec3u> &collision_mesh_face;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<FaceProp> &static_face_prop;
    const Vec<VertexParam> &dyn_vert_params;
    const Vec<FaceParam> &static_face_params;
    unsigned vertex_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        const Vec3u &f = collision_mesh_face[index];
        const Vec3f &t0 = collision_mesh_vertex[f[0]];
        const Vec3f &t1 = collision_mesh_vertex[f[1]];
        const Vec3f &t2 = collision_mesh_vertex[f[2]];
        const Vec3f &p0 = x0[vertex_index];
        const Vec3f &p1 = x1[vertex_index];
        const VertexParam &dyn_vparam =
            dyn_vert_params[dyn_vert_prop[vertex_index].param_index];
        const FaceParam &static_fparam =
            static_face_params[static_face_prop[index].param_index];
        float offset = dyn_vparam.offset + static_fparam.offset;
        float result = accd::point_triangle_ccd(p0, p1, t0, t1, t2, t0, t1, t2,
                                                offset, param);
        if (result < param.line_search_max_t) {
            toi = fminf(toi, result);
            assert(toi > 0.0f);
            return true;
        }
        return false;
    }
};

struct CollisionMeshPointFaceCCD_C2M {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<FaceProp> &dyn_face_prop;
    const Vec<VertexProp> &static_vert_prop;
    const Vec<FaceParam> &dyn_face_params;
    const Vec<VertexParam> &static_vert_params;
    unsigned vertex_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        if (!dyn_face_prop[index].fixed) {
            const Vec3u &f = face[index];
            const Vec3f &t00 = x0[f[0]];
            const Vec3f &t01 = x0[f[1]];
            const Vec3f &t02 = x0[f[2]];
            const Vec3f &t10 = x1[f[0]];
            const Vec3f &t11 = x1[f[1]];
            const Vec3f &t12 = x1[f[2]];
            const Vec3f &p = collision_mesh_vertex[vertex_index];
            const FaceParam &dyn_fparam =
                dyn_face_params[dyn_face_prop[index].param_index];
            const VertexParam &static_vparam =
                static_vert_params[static_vert_prop[vertex_index].param_index];
            float offset = dyn_fparam.offset + static_vparam.offset;
            float result = accd::point_triangle_ccd(p, p, t00, t01, t02, t10,
                                                    t11, t12, offset, param);
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

struct CollisionMeshEdgeEdgeCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec2u> &edge;
    const Vec<Vec2u> &collision_mesh_edge;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<EdgeProp> &dyn_edge_prop;
    const Vec<EdgeProp> &static_edge_prop;
    const Vec<EdgeParam> &dyn_edge_params;
    const Vec<EdgeParam> &static_edge_params;
    unsigned edge_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        if (!dyn_edge_prop[edge_index].fixed) {
            const Vec2u &e0 = edge[edge_index];
            const Vec2u &e1 = collision_mesh_edge[index];
            const Vec3f &p00 = x0[e0[0]];
            const Vec3f &p01 = x0[e0[1]];
            const Vec3f &p10 = x1[e0[0]];
            const Vec3f &p11 = x1[e0[1]];
            const Vec3f &q0 = collision_mesh_vertex[e1[0]];
            const Vec3f &q1 = collision_mesh_vertex[e1[1]];
            const EdgeParam &dyn_eparam =
                dyn_edge_params[dyn_edge_prop[edge_index].param_index];
            const EdgeParam &static_eparam =
                static_edge_params[static_edge_prop[index].param_index];
            float offset = dyn_eparam.offset + static_eparam.offset;
            float result = accd::edge_edge_ccd(p00, p01, q0, q1, p10, p11, q0,
                                               q1, offset, param);
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

struct PointFaceCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<VertexProp> &vertex_prop;
    const Vec<FaceProp> &face_prop;
    const Vec<VertexParam> &vertex_params;
    const Vec<FaceParam> &face_params;
    unsigned vertex_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        const Vec3u &f = face[index];
        bool either_dyn = vertex_prop[vertex_index].fix_index == 0 ||
                          face_prop[index].fixed == false;
        if (either_dyn) {
            int dup_i = -1;
            for (int i = 0; i < 3; ++i) {
                if (f[i] == vertex_index) {
                    dup_i = i;
                    break;
                }
            }
            float result = param.line_search_max_t;
            if (dup_i == -1) {
                const VertexParam &vparam =
                    vertex_params[vertex_prop[vertex_index].param_index];
                const FaceParam &fparam =
                    face_params[face_prop[index].param_index];
                float offset = vparam.offset + fparam.offset;
                const Vec3f &p0 = x0[vertex_index];
                const Vec3f &p1 = x1[vertex_index];
                const Vec3f &t00 = x0[f[0]];
                const Vec3f &t01 = x0[f[1]];
                const Vec3f &t02 = x0[f[2]];
                const Vec3f &t10 = x1[f[0]];
                const Vec3f &t11 = x1[f[1]];
                const Vec3f &t12 = x1[f[2]];
                result = accd::point_triangle_ccd(p0, p1, t00, t01, t02, t10,
                                                  t11, t12, offset, param);
            } else {
                const VertexParam &vparam =
                    vertex_params[vertex_prop[vertex_index].param_index];
                float offset = 2.0f * vparam.offset;
                unsigned i = dup_i;
                unsigned j = (i + 1) % 3;
                unsigned k = (i + 2) % 3;
                const Vec3f &p0 = x0[f[i]];
                const Vec3f &p1 = x1[f[i]];
                const Vec3f &q00 = x0[f[j]];
                const Vec3f &q10 = x1[f[j]];
                const Vec3f &q01 = x0[f[k]];
                const Vec3f &q11 = x1[f[k]];
                result = accd::point_edge_ccd(p0, p1, q00, q01, q10, q11,
                                              offset, param);
            }
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

struct EdgeEdgeCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec2u> &edge;
    const Vec<EdgeProp> &edge_prop;
    const Vec<EdgeParam> &edge_params;
    unsigned edge_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        bool either_dyn = edge_prop[edge_index].fixed == false ||
                          edge_prop[index].fixed == false;
        if (edge_index < index && either_dyn) {
            const Vec2u &e0 = edge[edge_index];
            const Vec2u &e1 = edge[index];
            float result = param.line_search_max_t;
            const EdgeParam &eparam_edge =
                edge_params[edge_prop[edge_index].param_index];
            const EdgeParam &eparam_index =
                edge_params[edge_prop[index].param_index];
            float offset = eparam_edge.offset + eparam_index.offset;
            if (!edge_has_shared_vert(e0, e1)) {
                const Vec3f &p00 = x0[e0[0]];
                const Vec3f &p01 = x0[e0[1]];
                const Vec3f &q00 = x0[e1[0]];
                const Vec3f &q01 = x0[e1[1]];
                const Vec3f &p10 = x1[e0[0]];
                const Vec3f &p11 = x1[e0[1]];
                const Vec3f &q10 = x1[e1[0]];
                const Vec3f &q11 = x1[e1[1]];
                result = accd::edge_edge_ccd(p00, p01, q00, q01, p10, p11, q10,
                                             q11, offset, param);
            } else {
                const Vec2u ij[] = {Vec2u(0, 0), Vec2u(0, 1), Vec2u(1, 0),
                                    Vec2u(1, 1)};
                for (unsigned k = 0; k < 4; ++k) {
                    unsigned i = ij[k][0];
                    unsigned j = ij[k][1];
                    if (e0[i] == e1[j]) {
                        unsigned idx0 = e0[i];
                        unsigned idx1 = e0[1 - i];
                        unsigned idx2 = e1[1 - j];
                        const Vec3f &q00 = x0[idx0];
                        const Vec3f &q10 = x1[idx0];
                        const Vec3f &q01 = x0[idx1];
                        const Vec3f &q11 = x1[idx1];
                        const Vec3f &q02 = x0[idx2];
                        const Vec3f &q12 = x1[idx2];
                        float toi_0 = accd::point_edge_ccd(
                            q01, q11, q00, q02, q10, q12, offset, param);
                        float toi_1 = accd::point_edge_ccd(
                            q02, q12, q00, q01, q10, q11, offset, param);
                        result = fminf(toi_0, toi_1);
                        break;
                    }
                }
            }
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

__device__ void vertex_constraint_line_search(const DataSet &data,
                                              const Vec<Vec3f> &y0,
                                              const Vec<Vec3f> &y1,
                                              Vec<float> toi_vert,
                                              ParamSet param, unsigned i) {
    const Vec3f x1 = param.line_search_max_t * (y1[i] - y0[i]) + y0[i];
    const Vec3f &x0 = y0[i];
    const VertexProp &prop = data.prop.vertex[i];
    if (prop.fix_index > 0) {
        const FixPair &fix = data.constraint.fix[prop.fix_index - 1];
        if (fix.kinematic == false) {
            const Vec3f &position = fix.position;
            float r0 = (x0 - position).norm();
            float r1 = (x1 - position).norm();
            assert(r0 < fix.ghat);
            float r = fix.ghat;
            if (r1 > r) {
                // (1.0f - t) r0 + t r1 = r
                // r0 - t r0 + t r1 = r
                // t (r1 - r0) = r - r0
                // t = (r - r0) / (r1 - r0)
                float denom = r1 - r0;
                if (denom) {
                    float t = (r - r0) / denom;
                    toi_vert[i] =
                        fminf(toi_vert[i], param.line_search_max_t * t);
                    assert(toi_vert[i] > 0.0f);
                }
            }
        }
    } else {
        for (unsigned j = 0; j < data.constraint.sphere.size; ++j) {
            const Sphere &sphere = data.constraint.sphere[j];
            if (!sphere.kinematic) {
                bool reverse = sphere.reverse;
                bool bowl = sphere.bowl;
                const Vec3f &center = sphere.center;
                const Vec3f center0 = (bowl && (x0[1] > center[1]))
                                          ? Vec3f(center[0], x0[1], center[2])
                                          : center;
                const Vec3f center1 = (bowl && (x1[1] > center[1]))
                                          ? Vec3f(center[0], x1[1], center[2])
                                          : center;
                float r = sphere.radius;
                float r0 = (x0 - center0).norm();
                float r1 = (x1 - center1).norm();
                if (reverse) {
                    assert(r0 < r);
                } else {
                    assert(r0 > r);
                }
                bool intersected = (r0 - r) * (r1 - r) <= 0.0f;
                if (intersected) {
                    // (1.0f - t) r0 + t r1 = r
                    // r0 - t r0 + t r1 = r
                    // t (r1 - r0) = radius - r0
                    // t = (r - r0) / (r1 - r0)
                    float t = (r - r0) / (r1 - r0);
                    toi_vert[i] =
                        fminf(toi_vert[i], param.line_search_max_t * t);
                    assert(toi_vert[i] > 0.0f);
                }
            }
        }

        for (unsigned j = 0; j < data.constraint.floor.size; ++j) {
            const Floor &floor = data.constraint.floor[j];
            if (!floor.kinematic) {
                const Vec3f &up = floor.up;
                const Vec3f &ground = floor.ground;
                float h0 = up.dot(x0 - ground);
                float h1 = up.dot(x1 - ground);
                assert(h0 >= 0.0f);
                if (h1 < 0.0f) {
                    // (1.0f - t) h0 + t h1 = 0
                    // h0 - t h0 + t h1 = 0
                    // t (h1 - h0) = - h0
                    // t = - h0 / (h1 - h0)
                    float t = -h0 / (h1 - h0);
                    toi_vert[i] =
                        fminf(toi_vert[i], param.line_search_max_t * t);
                    assert(toi_vert[i] > 0.0f);
                }
            }
        }
    }
}

float line_search(const DataSet &data, const Vec<Vec3f> &x0,
                  const Vec<Vec3f> &x1, const ParamSet &param) {

    const MeshInfo &mesh = data.mesh;
    const BVHSet &bvhset = data.bvh;

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned edge_count = mesh.mesh.edge.size;
    unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
    unsigned max_contact_vert = (surface_vert_count > collision_mesh_vert_count)
                                    ? surface_vert_count
                                    : collision_mesh_vert_count;

    const BVH &face_bvh = bvhset.face;
    const BVH &edge_bvh = bvhset.edge;
    const BVH &collision_mesh_face_bvh = data.constraint.mesh.face_bvh;
    const BVH &collision_mesh_edge_bvh = data.constraint.mesh.edge_bvh;
    const Vec<Vec2u> &collision_mesh_edge = data.constraint.mesh.edge;
    const Vec<Vec3u> &collision_mesh_face = data.constraint.mesh.face;
    const Vec<Vec3f> &collision_mesh_vertex = data.constraint.mesh.vertex;
    const Vec<AABB> face_aabb = storage::face_aabb;
    const Vec<AABB> edge_aabb = storage::edge_aabb;
    const Vec<AABB> collision_mesh_face_aabb =
        storage::collision_mesh_face_aabb;
    const Vec<AABB> collision_mesh_edge_aabb =
        storage::collision_mesh_edge_aabb;

    buffer::MemoryPool &pool = buffer::get();
    auto intersection_flag = pool.get<char>(edge_count);
    auto toi_vtf = pool.get<float>(max_contact_vert);
    auto toi_ee = pool.get<float>(edge_count);

    intersection_flag.clear(0);
    toi_vtf.clear(param.line_search_max_t);
    toi_ee.clear(param.line_search_max_t);

    Vec<char> intersection_flag_vec = intersection_flag.as_vec();
    Vec<float> toi_vtf_vec = toi_vtf.as_vec();
    Vec<float> toi_ee_vec = toi_ee.as_vec();

    Vec<VertexParam> vertex_params = data.param_arrays.vertex;
    Vec<EdgeParam> edge_params = data.param_arrays.edge;
    Vec<FaceParam> face_params = data.param_arrays.face;
    Vec<VertexParam> collision_vertex_params =
        data.constraint.mesh.param_arrays.vertex;
    Vec<EdgeParam> collision_edge_params =
        data.constraint.mesh.param_arrays.edge;
    Vec<FaceParam> collision_face_params =
        data.constraint.mesh.param_arrays.face;

    DISPATCH_START(surface_vert_count)
    [data, x0, x1, toi_vtf_vec, param] __device__(unsigned i) mutable {
        vertex_constraint_line_search(data, x0, x1, toi_vtf_vec, param, i);
    } DISPATCH_END;

    if (!param.disable_contact) {
        DISPATCH_START(surface_vert_count)
        [data, mesh, x0, x1, face_bvh, face_aabb, toi_vtf_vec, vertex_params,
         face_params, param] __device__(unsigned i) mutable {
            const VertexParam &vparam =
                vertex_params[data.prop.vertex[i].param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            float toi = param.line_search_max_t;
            PointFaceCCD ccd = {x0,
                                x1,
                                mesh.mesh.face,
                                data.prop.vertex,
                                data.prop.face,
                                vertex_params,
                                face_params,
                                i,
                                toi,
                                param};
            AABB_AABB_Tester<PointFaceCCD> op(ccd);
            AABB aabb =
                aabb::make(x0[i], toi * (x1[i] - x0[i]) + x0[i], ext_eps);
            aabb::query(face_bvh, face_aabb, op, aabb);
            toi_vtf_vec[i] = fmin(toi_vtf_vec[i], toi);
        } DISPATCH_END;

        DISPATCH_START(surface_vert_count)
        [data, mesh, x0, x1, collision_mesh_face, collision_mesh_face_bvh,
         collision_mesh_face_aabb, collision_mesh_vertex, toi_vtf_vec,
         vertex_params, collision_face_params,
         param] __device__(unsigned i) mutable {
            if (data.prop.vertex[i].fix_index == 0) {
                const VertexParam &vparam =
                    vertex_params[data.prop.vertex[i].param_index];
                float ext_eps = 0.5f * vparam.ghat + vparam.offset;
                float toi = param.line_search_max_t;
                CollisionMeshPointFaceCCD_M2C ccd = {
                    x0,
                    x1,
                    mesh.mesh.face,
                    collision_mesh_face,
                    collision_mesh_vertex,
                    data.prop.vertex,
                    data.constraint.mesh.prop.face,
                    vertex_params,
                    collision_face_params,
                    i,
                    toi,
                    param};
                AABB_AABB_Tester<CollisionMeshPointFaceCCD_M2C> op(ccd);
                AABB aabb =
                    aabb::make(x0[i], toi * (x1[i] - x0[i]) + x0[i], ext_eps);
                aabb::query(collision_mesh_face_bvh, collision_mesh_face_aabb,
                            op, aabb);
                toi_vtf_vec[i] = fmin(toi_vtf_vec[i], toi);
            }
        } DISPATCH_END;

        unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
        DISPATCH_START(collision_mesh_vert_count)
        [data, mesh, x0, x1, collision_mesh_face, collision_mesh_vertex,
         toi_vtf_vec, face_bvh, face_aabb, face_params, collision_vertex_params,
         param] __device__(unsigned i) mutable {
            float toi = param.line_search_max_t;
            CollisionMeshPointFaceCCD_C2M ccd = {
                x0,
                x1,
                mesh.mesh.face,
                collision_mesh_vertex,
                data.prop.face,
                data.constraint.mesh.prop.vertex,
                face_params,
                collision_vertex_params,
                i,
                toi,
                param};
            AABB_AABB_Tester<CollisionMeshPointFaceCCD_C2M> op(ccd);
            Vec3f q = collision_mesh_vertex[i];
            const VertexParam &vparam =
                collision_vertex_params[data.constraint.mesh.prop.vertex[i]
                                            .param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            aabb::query(face_bvh, face_aabb, op, aabb::make(q, ext_eps));
            toi_vtf_vec[i] = fmin(toi_vtf_vec[i], toi);
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, mesh, x0, x1, edge_bvh, edge_aabb, toi_ee_vec, edge_params,
         param] __device__(unsigned i) mutable {
            Vec2u edge = mesh.mesh.edge[i];
            float toi = param.line_search_max_t;
            const EdgeParam &eparam =
                edge_params[data.prop.edge[i].param_index];
            float ext_eps = 0.5f * eparam.ghat + eparam.offset;
            AABB aabb0 = aabb::make(x0[edge[0]], x0[edge[1]], ext_eps);
            AABB aabb1 = aabb::make(
                toi * (x1[edge[0]] - x0[edge[0]]) + x0[edge[0]],
                toi * (x1[edge[1]] - x0[edge[1]]) + x0[edge[1]], ext_eps);
            AABB aabb = aabb::join(aabb0, aabb1);
            EdgeEdgeCCD ccd = {
                x0, x1,  mesh.mesh.edge, data.prop.edge, edge_params,
                i,  toi, param};
            AABB_AABB_Tester<EdgeEdgeCCD> op(ccd);
            aabb::query(edge_bvh, edge_aabb, op, aabb);
            toi_ee_vec[i] = fmin(toi, toi_ee_vec[i]);
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, mesh, x0, x1, collision_mesh_edge_bvh, collision_mesh_edge,
         collision_mesh_vertex, collision_mesh_edge_aabb, toi_ee_vec,
         edge_params, collision_edge_params,
         param] __device__(unsigned i) mutable {
            Vec2u edge = mesh.mesh.edge[i];
            float toi = param.line_search_max_t;
            const EdgeParam &eparam =
                edge_params[data.prop.edge[i].param_index];
            float ext_eps = 0.5f * eparam.ghat + eparam.offset;
            AABB aabb0 = aabb::make(x0[edge[0]], x0[edge[1]], ext_eps);
            AABB aabb1 = aabb::make(
                toi * (x1[edge[0]] - x0[edge[0]]) + x0[edge[0]],
                toi * (x1[edge[1]] - x0[edge[1]]) + x0[edge[1]], ext_eps);
            AABB aabb = aabb::join(aabb0, aabb1);
            CollisionMeshEdgeEdgeCCD ccd = {x0,
                                            x1,
                                            mesh.mesh.edge,
                                            collision_mesh_edge,
                                            collision_mesh_vertex,
                                            data.prop.edge,
                                            data.constraint.mesh.prop.edge,
                                            edge_params,
                                            collision_edge_params,
                                            i,
                                            toi,
                                            param};

            AABB_AABB_Tester<CollisionMeshEdgeEdgeCCD> op(ccd);
            aabb::query(collision_mesh_edge_bvh, collision_mesh_edge_aabb, op,
                        aabb);
            toi_ee_vec[i] = fmin(toi, toi_ee_vec[i]);
        } DISPATCH_END;
    }

    float toi = fminf(kernels::min_array(toi_vtf_vec.data, toi_vtf_vec.size,
                                         param.line_search_max_t),
                      kernels::min_array(toi_ee_vec.data, toi_ee_vec.size,
                                         param.line_search_max_t));

    return toi / param.line_search_max_t;
}

template <class T, class Y>
__device__ bool point_triangle_inside(const Vec3<T> &p, const Vec3<T> &t0,
                                      const Vec3<T> &t1, const Vec3<T> &t2) {
    Vec3<Y> r0 = (t1 - t0).template cast<Y>();
    Vec3<Y> r1 = (t2 - t0).template cast<Y>();
    Mat3x2<Y> a;
    a << r0, r1;
    Eigen::Transpose<Mat3x2<Y>> a_t = a.transpose();
    Y det;
    Vec2<Y> c;
    distance::solve<Y>(a_t * a, a_t * (p - t0).template cast<Y>(), c, det);
    if (det) {
        Vec3<Y> w = Vec3<Y>(det - c[0] - c[1], c[0], c[1]) / det;
        return w.minCoeff() >= 0.0f && w.maxCoeff() <= 1.0f;
    } else {
        return false;
    }
}

__device__ bool edge_triangle_intersect(const Vec3f &_e0, const Vec3f &_e1,
                                        const Vec3f &_x0, const Vec3f &_x1,
                                        const Vec3f &_x2) {
    Vec3f n = (_x1 - _x0).cross(_x2 - _x0);
    float s1 = (_e0 - _x0).dot(n);
    float s2 = (_e1 - _x0).dot(n);
    if (s1 * s2 < 0.0f) {
        float det = s1 - s2;
        if (det) {
            Vec3f r = (_e1 - _e0) * s1 / det;
            Vec3f t0 = _x0 - _e0;
            Vec3f t1 = _x1 - _e0;
            Vec3f t2 = _x2 - _e0;
            return point_triangle_inside<float, float>(r, t0, t1, t2);
        }
    }
    return false;
}

class EdgeEdgeIntersectTester {
  public:
    __device__
    EdgeEdgeIntersectTester(const Vec<EdgeProp> &prop,
                            const Vec<EdgeParam> &edge_params,
                            const Vec<Vec3f> &vertex, const Vec<Vec2u> &edge,
                            const ParamSet &param, unsigned edge_index)
        : prop(prop), edge_params(edge_params), vertex(vertex), edge(edge),
          param(param), edge_index(edge_index) {}
    __device__ bool operator()(unsigned index) {
        if (index < edge_index) {
            const Vec2u &e0 = edge[edge_index];
            const Vec2u &e1 = edge[index];
            bool either_dyn =
                prop[edge_index].fixed == false || prop[index].fixed == false;
            if (either_dyn) {
                const EdgeParam &eparam_edge =
                    edge_params[prop[edge_index].param_index];
                const EdgeParam &eparam_index =
                    edge_params[prop[index].param_index];
                float offset = eparam_edge.offset + eparam_index.offset;
                if (!edge_has_shared_vert(e0, e1)) {
                    Vec3f p0 = vertex[e0[0]];
                    Vec3f p1 = vertex[e0[1]];
                    Vec3f q0 = vertex[e1[0]];
                    Vec3f q1 = vertex[e1[1]];
                    Vec4f c = distance::edge_edge_distance_coeff_unclassified<
                        float, float>(p0, p1, q0, q1);
                    Vec3f x0 = c[0] * p0 + c[1] * p1;
                    Vec3f x1 = c[2] * q0 + c[3] * q1;
                    Vec3f e = x0 - x1;
                    if (e.dot(e) < offset * offset) {
                        return true;
                    }
                } else {
                    const Vec2u ij[] = {Vec2u(0, 0), Vec2u(0, 1), Vec2u(1, 0),
                                        Vec2u(1, 1)};
                    for (unsigned k = 0; k < 4; ++k) {
                        unsigned i = ij[k][0];
                        unsigned j = ij[k][1];
                        if (e0[i] == e1[j]) {
                            unsigned idx0 = e0[i];
                            unsigned idx1 = e0[1 - i];
                            unsigned idx2 = e1[1 - j];
                            const Vec3f &q0 = vertex[idx0];
                            const Vec3f &q1 = vertex[idx1];
                            const Vec3f &q2 = vertex[idx2];
                            Vec2f c_0 = distance::
                                point_edge_distance_coeff_unclassified<float,
                                                                       float>(
                                    q1, q0, q2);
                            Vec2f c_1 = distance::
                                point_edge_distance_coeff_unclassified<float,
                                                                       float>(
                                    q2, q0, q1);
                            Vec3f e_0 = (c_0[0] * q0 + c_0[1] * q2) - q1;
                            Vec3f e_1 = (c_1[0] * q0 + c_1[1] * q1) - q2;
                            float sqr_d0 = e_0.dot(e_0);
                            float sqr_d1 = e_1.dot(e_1);
                            if (std::min(sqr_d0, sqr_d1) < offset * offset) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }
    const Vec<EdgeProp> &prop;
    const Vec<EdgeParam> &edge_params;
    const Vec<Vec3f> &vertex;
    const Vec<Vec2u> &edge;
    const ParamSet &param;
    unsigned edge_index;
};

class FaceEdgeIntersectTester {
  public:
    __device__
    FaceEdgeIntersectTester(const Vec<FaceProp> &face_prop,
                            const Vec<EdgeProp> &edge_prop,
                            const Vec<Vec3f> &vertex, const Vec<Vec3u> &face,
                            const Vec<Vec2u> &edge, unsigned edge_index)
        : face_prop(face_prop), edge_prop(edge_prop), vertex(vertex),
          face(face), edge(edge), edge_index(edge_index) {}
    __device__ bool operator()(unsigned index) {
        bool either_dyn = face_prop[index].fixed == false ||
                          edge_prop[edge_index].fixed == false;
        if (either_dyn) {
            Vec3u f = face[index];
            unsigned e0 = edge[edge_index][0];
            unsigned e1 = edge[edge_index][1];
            if (e0 != f[0] && e0 != f[1] && e0 != f[2] && //
                e1 != f[0] && e1 != f[1] && e1 != f[2]) {
                const Vec3f &x0 = vertex[f[0]];
                const Vec3f &x1 = vertex[f[1]];
                const Vec3f &x2 = vertex[f[2]];
                const Vec3f &y0 = vertex[e0];
                const Vec3f &y1 = vertex[e1];
                if (edge_triangle_intersect(y0, y1, x0, x1, x2)) {
                    return true;
                }
            }
        }
        return false;
    }
    const Vec<FaceProp> &face_prop;
    const Vec<EdgeProp> &edge_prop;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3u> &face;
    const Vec<Vec2u> &edge;
    unsigned edge_index;
};

class CollisionMeshFaceEdgeIntersectTester {
  public:
    __device__ CollisionMeshFaceEdgeIntersectTester(const Vec<Vec3f> &vertex,
                                                    const Vec<Vec3u> &face,
                                                    const Vec3f &y0,
                                                    const Vec3f &y1)
        : vertex(vertex), face(face), y0(y0), y1(y1) {}
    __device__ bool operator()(unsigned index) {
        Vec3u f = face[index];
        const Vec3f &x0 = vertex[f[0]];
        const Vec3f &x1 = vertex[f[1]];
        const Vec3f &x2 = vertex[f[2]];
        if (edge_triangle_intersect(y0, y1, x0, x1, x2)) {
            return true;
        }
        return false;
    }
    const Vec<Vec3f> &vertex;
    const Vec<Vec3u> &face;
    Vec3f y0, y1;
};

bool check_intersection(const DataSet &data, const Vec<Vec3f> &vertex,
                        const ParamSet &param) {

    unsigned edge_count = data.mesh.mesh.edge.size;
    const BVH &face_bvh = data.bvh.face;
    const BVH &edge_bvh = data.bvh.edge;
    const BVH &collision_mesh_face_bvh = data.constraint.mesh.face_bvh;
    const MeshInfo &mesh = data.mesh;
    const Vec<Vec3u> &collision_mesh_face = data.constraint.mesh.face;
    const Vec<AABB> face_aabb = storage::face_aabb;
    const Vec<AABB> edge_aabb = storage::edge_aabb;
    const Vec<AABB> collision_mesh_face_aabb =
        storage::collision_mesh_face_aabb;

    buffer::MemoryPool &pool = buffer::get();
    auto intersection_flag = pool.get<char>(edge_count);
    intersection_flag.clear(0);
    const Vec<Vec3f> &collision_mesh_vertex = data.constraint.mesh.vertex;

    Vec<char> intersection_flag_vec = intersection_flag.as_vec();
    Vec<EdgeParam> edge_params = data.param_arrays.edge;

    DISPATCH_START(edge_count)
    [data, mesh, vertex, collision_mesh_vertex, face_bvh, face_aabb, edge_bvh,
     edge_aabb, collision_mesh_face_bvh, collision_mesh_face_aabb,
     collision_mesh_face, intersection_flag_vec, edge_params,
     param] __device__(unsigned i) mutable {
        const Vec2u &edge = mesh.mesh.edge[i];
        Vec3f y0 = vertex[edge[0]];
        Vec3f y1 = vertex[edge[1]];
        AABB aabb = aabb::make(y0, y1, 0.0f);
        FaceEdgeIntersectTester tester_0(data.prop.face, data.prop.edge, vertex,
                                         mesh.mesh.face, mesh.mesh.edge, i);
        AABB_AABB_Tester<FaceEdgeIntersectTester> op_0(tester_0);
        if (aabb::query(face_bvh, face_aabb, op_0, aabb)) {
            intersection_flag_vec[i] = 1;
        }
        EdgeEdgeIntersectTester tester_1(data.prop.edge, edge_params, vertex,
                                         mesh.mesh.edge, param, i);
        AABB_AABB_Tester<EdgeEdgeIntersectTester> op_1(tester_1);
        if (aabb::query(edge_bvh, edge_aabb, op_1, aabb)) {
            intersection_flag_vec[i] = 1;
        }
        CollisionMeshFaceEdgeIntersectTester tester_2(
            collision_mesh_vertex, collision_mesh_face, y0, y1);
        AABB_AABB_Tester<CollisionMeshFaceEdgeIntersectTester> op_2(tester_2);
        if (aabb::query(collision_mesh_face_bvh, collision_mesh_face_aabb, op_2,
                        aabb)) {
            intersection_flag_vec[i] = 1;
        }
    } DISPATCH_END;

    bool result = kernels::max_array(intersection_flag_vec.data, edge_count,
                                     char(0)) == 0;
    return result;
}

} // namespace contact
