// File: contact.cu
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CONTACT_HPP
#define CONTACT_HPP

#include "../aabb/aabb.hpp"
#include "../barrier/barrier.hpp"
#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../energy/model/fix.hpp"
#include "../energy/model/friction.hpp"
#include "../energy/model/push.hpp"
#include "../math/distance.hpp"
#include "../simplelog/SimpleLog.h"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "accd.hpp"
#include "contact.hpp"

namespace contact {

namespace storage {
Vec<char> intersection_flag;
Vec<unsigned> num_contact_vertex_face;
Vec<unsigned> num_contact_edge_edge;
Vec<AABB> face_aabb, edge_aabb, vertex_aabb;
Vec<AABB> collision_mesh_face_aabb, collision_mesh_edge_aabb;
Vec<float> edge_edge_toi;
Vec<float> vertex_face_toi;
Vec<float> contact_force;
} // namespace storage

static unsigned max_of_two(unsigned a, unsigned b) { return a > b ? a : b; }
__device__ inline float sqr(float x) { return x * x; }

void initialize(const DataSet &data, const ParamSet &param) {
    unsigned surface_vert_count = data.surface_vert_count;
    unsigned colllision_mesh_vert_count =
        data.constraint.mesh.active ? data.constraint.mesh.vertex.size : 0;
    unsigned edge_count = data.mesh.mesh.edge.size;
    BVHSet bvhset = data.bvh;
    storage::intersection_flag = Vec<char>::alloc(edge_count).clear(0);
    storage::face_aabb =
        Vec<AABB>::alloc(bvhset.face.node.size, param.bvh_alloc_factor);
    storage::edge_aabb =
        Vec<AABB>::alloc(bvhset.edge.node.size, param.bvh_alloc_factor);
    storage::vertex_aabb =
        Vec<AABB>::alloc(bvhset.vertex.node.size, param.bvh_alloc_factor);
    if (data.constraint.mesh.active) {
        storage::collision_mesh_face_aabb =
            Vec<AABB>::alloc(data.constraint.mesh.face_bvh.node.size);
        storage::collision_mesh_edge_aabb =
            Vec<AABB>::alloc(data.constraint.mesh.edge_bvh.node.size);
    }
    storage::contact_force = Vec<float>::alloc(3 * surface_vert_count);
    storage::num_contact_vertex_face = Vec<unsigned>::alloc(
        max_of_two(surface_vert_count, colllision_mesh_vert_count));
    storage::num_contact_edge_edge = Vec<unsigned>::alloc(edge_count);
    storage::vertex_face_toi = Vec<float>::alloc(
        max_of_two(surface_vert_count, colllision_mesh_vert_count));
    storage::edge_edge_toi = Vec<float>::alloc(edge_count);
}

void resize_aabb(const BVHSet &bvh) {
    storage::face_aabb.resize(bvh.face.node.size);
    storage::edge_aabb.resize(bvh.edge.node.size);
}

__device__ float make_face_offset(const ParamSet &param) {
    return param.contact_offset;
}

__device__ float make_edge_offset(const Vec2u &edge,
                                  const Vec<VertexKinematic> &kinematic,
                                  const ParamSet &param) {
    if (kinematic[edge[0]].rod || kinematic[edge[1]].rod) {
        return param.rod_offset;
    } else {
        return param.contact_offset;
    }
}

__device__ float make_vertex_offset(const unsigned &index,
                                    const Vec<VertexKinematic> &kinematic,
                                    const ParamSet &param) {
    if (kinematic[index].rod) {
        return param.rod_offset;
    } else {
        return param.contact_offset;
    }
}

__device__ void update_face_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                                 const BVH &bvh, Vec<AABB> &aabb,
                                 const Vec<Vec3u> &face, unsigned level,
                                 const ParamSet &param, float ext_eps,
                                 unsigned i) {
    if (!ext_eps) {
        ext_eps = 0.5f * param.contact_ghat + make_face_offset(param);
    }
    unsigned index = bvh.level(level, i);
    Vec2u node = bvh.node[index];
    if (node[1] == 0) {
        unsigned leaf = node[0] - 1;
        aabb[index] =
            aabb::join(aabb::make(x0[face[leaf][0]], x0[face[leaf][1]],
                                  x0[face[leaf][2]], ext_eps),
                       aabb::make(x1[face[leaf][0]], x1[face[leaf][1]],
                                  x1[face[leaf][2]], ext_eps));
    } else {
        unsigned left = node[0] - 1;
        unsigned right = node[1] - 1;
        aabb[index] = aabb::join(aabb[left], aabb[right]);
    }
}

__device__ void update_edge_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                                 const BVH &bvh, Vec<AABB> aabb,
                                 const Vec<Vec2u> &edge, unsigned level,
                                 const ParamSet &param, float ext_eps,
                                 const Vec<VertexKinematic> &kinematic,
                                 unsigned i) {
    unsigned index = bvh.level(level, i);
    Vec2u node = bvh.node[index];
    if (node[1] == 0) {
        unsigned leaf = node[0] - 1;
        unsigned i0 = edge[leaf][0];
        unsigned i1 = edge[leaf][1];
        if (!ext_eps) {
            ext_eps = 0.5f * param.contact_ghat +
                      make_edge_offset(edge[leaf], kinematic, param);
            aabb[index] = aabb::join(aabb::make(x0[i0], x0[i1], ext_eps),
                                     aabb::make(x1[i0], x1[i1], ext_eps));
        } else {
            aabb[index] = aabb::join(aabb::make(x0[i0], x0[i1], ext_eps),
                                     aabb::make(x1[i0], x1[i1], ext_eps));
        }
    } else {
        unsigned left = node[0] - 1;
        unsigned right = node[1] - 1;
        aabb[index] = aabb::join(aabb[left], aabb[right]);
    }
}

__device__ void update_vertex_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                                   const BVH &bvh, Vec<AABB> aabb,
                                   unsigned level, const ParamSet &param,
                                   float ext_eps,
                                   const Vec<VertexKinematic> &kinematic,
                                   unsigned i) {
    unsigned index = bvh.level(level, i);
    Vec2u node = bvh.node[index];
    if (node[1] == 0) {
        unsigned leaf = node[0] - 1;
        if (!ext_eps) {
            ext_eps = 0.5f * param.contact_ghat +
                      make_vertex_offset(leaf, kinematic, param);
        }
        aabb[index] = aabb::make(x0[leaf], x1[leaf], ext_eps);
    } else {
        unsigned left = node[0] - 1;
        unsigned right = node[1] - 1;
        aabb[index] = aabb::join(aabb[left], aabb[right]);
    }
}

void update_aabb(const DataSet &host_data, const DataSet &dev_data,
                 const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                 const Vec<VertexKinematic> &kinematic, const ParamSet &param) {

    const MeshInfo &mesh = dev_data.mesh;
    const BVHSet &bvhset = dev_data.bvh;

    const BVH &face_bvh = bvhset.face;
    Vec<AABB> &face_aabb = storage::face_aabb;
    for (unsigned level = 0; level < bvhset.face.level.size; ++level) {
        unsigned count = host_data.bvh.face.level.count(level);
        DISPATCH_START(count)
        [mesh, param, x0, x1, face_bvh, level,
         face_aabb] __device__(unsigned i) mutable {
            update_face_aabb(x0, x1, face_bvh, face_aabb, mesh.mesh.face, level,
                             param, 0.0f, i);
        } DISPATCH_END;
    }
    const BVH &edge_bvh = bvhset.edge;
    Vec<AABB> &edge_aabb = storage::edge_aabb;
    for (unsigned level = 0; level < bvhset.edge.level.size; ++level) {
        unsigned count = host_data.bvh.edge.level.count(level);
        DISPATCH_START(count)
        [mesh, kinematic, param, x0, x1, edge_bvh, level,
         edge_aabb] __device__(unsigned i) mutable {
            update_edge_aabb(x0, x1, edge_bvh, edge_aabb, mesh.mesh.edge, level,
                             param, 0.0f, kinematic, i);
        } DISPATCH_END;
    }
    const BVH &vertex_bvh = bvhset.vertex;
    Vec<AABB> &vertex_aabb = storage::vertex_aabb;
    for (unsigned level = 0; level < bvhset.vertex.level.size; ++level) {
        unsigned count = host_data.bvh.vertex.level.count(level);
        DISPATCH_START(count)
        [mesh, kinematic, param, x0, x1, vertex_bvh, level,
         vertex_aabb] __device__(unsigned i) mutable {
            update_vertex_aabb(x0, x1, vertex_bvh, vertex_aabb, level, param,
                               0.0f, kinematic, i);
        } DISPATCH_END;
    }
}

void update_collision_mesh_aabb(const DataSet &host_data,
                                const DataSet &dev_data,
                                const ParamSet &param) {
    if (host_data.constraint.mesh.active) {
        const Vec<Vec3f> &vertex = dev_data.constraint.mesh.vertex;
        const BVH &face_bvh = dev_data.constraint.mesh.face_bvh;
        Vec<AABB> &face_aabb = storage::collision_mesh_face_aabb;
        float margin = 0.5f * param.contact_ghat;
        const Vec<Vec3u> &face = dev_data.constraint.mesh.face;
        for (unsigned level = 0; level < face_bvh.level.size; ++level) {
            unsigned count =
                host_data.constraint.mesh.face_bvh.level.count(level);
            DISPATCH_START(count)
            [vertex, face_bvh, param, level, face_aabb, margin,
             face] __device__(unsigned i) mutable {
                update_face_aabb(vertex, vertex, face_bvh, face_aabb, face,
                                 level, param, margin, i);
            } DISPATCH_END;
        }
        const BVH &edge_bvh = dev_data.constraint.mesh.edge_bvh;
        Vec<AABB> &edge_aabb = storage::collision_mesh_edge_aabb;
        const Vec<Vec2u> &edge = dev_data.constraint.mesh.edge;
        for (unsigned level = 0; level < edge_bvh.level.size; ++level) {
            unsigned count =
                host_data.constraint.mesh.edge_bvh.level.count(level);
            Vec<VertexKinematic> kinematic;
            kinematic.size = 0;
            DISPATCH_START(count)
            [vertex, edge_bvh, kinematic, param, level, edge_aabb, margin,
             edge] __device__(unsigned i) mutable {
                update_edge_aabb(vertex, vertex, edge_bvh, edge_aabb, edge,
                                 level, param, margin, kinematic, i);
            } DISPATCH_END;
        }
    }
}

__device__ bool intersect_free(unsigned a, unsigned b, unsigned c, unsigned d) {
    return a != b && a != c && a != d && b != c && b != d && c != d;
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
    const Vec<VertexProp> &prop, Barrier barrier, float dt, unsigned count,
    const ParamSet &param, int stage, unsigned i) {

    SVecf<N> mass;
    Vec3f ex0 = Vec3f::Zero();
    Vec3f ex = Vec3f::Zero();
    float area = 0.0f;
    float wsum = 0.0f;
    for (int ii = 0; ii < N; ++ii) {
        unsigned index = prox.index[ii];
        mass[ii] = prop[index].mass;
        float w = prox.value[ii];
        area += fabsf(w) * prop[index].area;
        wsum += fabsf(w);
        ex0 += w * x0[index];
        ex += w * x[index];
    }
    Vec3f dx = ex - ex0;
    Vec3f du = dx / dt;
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

        Friction friction(f, ex - ex0, normal, param.friction,
                          param.friction_eps);
        f += friction.gradient();
        H += friction.hessian();

        SMatf<3, N> ext_force;
        SMatf<N * 3, N * 3> ext_hess;
        extend_contact_force_hess<N>(prox, f, H, ext_force, ext_hess);
        utility::atomic_embed_force<N>(prox.index, count * ext_force,
                                       force_out);
        atomic_embed_hessian<N>(prox.index, count * ext_hess, fixed_out,
                                dyn_out);
    }
}

inline __device__ bool check_kinematic_pair(bool a, bool b) {
    if (a && b) {
        return false;
    } else {
        return true;
    }
}

struct PointPointContactForceHessEmbed {

    unsigned vertex_index;
    unsigned rod_count;
    const Kinematic &kinematic;
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
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (index < vertex_index &&
            check_kinematic_pair(kinematic.vertex[vertex_index].active,
                                 kinematic.vertex[index].active)) {
            const Vec3f &x = eval_x[vertex_index];
            const Vec3f &y = eval_x[index];
            Vec3f e = x - y;
            float offset =
                make_vertex_offset(index, kinematic.vertex, param) +
                make_vertex_offset(vertex_index, kinematic.vertex, param);
            float ghat_sqr = sqr(param.contact_ghat + offset);
            float sqr_dist = e.squaredNorm();
            if (sqr_dist < ghat_sqr) {
                unsigned count = 0;
                if (neighbor.edge.nnz) {
                    if (neighbor.edge.count(index)) {
                        for (unsigned j = neighbor.edge.offset[index];
                             j < neighbor.edge.offset[index + 1]; ++j) {
                            Vec2u f = edge[neighbor.edge.data[j]];
                            Vec2f c = distance::point_edge_distance_coeff(
                                x, eval_x[f[0]], eval_x[f[1]]);
                            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                                continue;
                            } else {
                                ++count;
                            }
                        }
                        if (neighbor.face.nnz) {
                            for (unsigned j = neighbor.face.offset[index];
                                 j < neighbor.face.offset[index + 1]; ++j) {
                                Vec3u f = face[neighbor.face.data[j]];
                                Vec3f c =
                                    distance::point_triangle_distance_coeff(
                                        x, eval_x[f[0]], eval_x[f[1]],
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
                        count = 1;
                    }
                } else {
                    count = 1;
                }
                if (count) {
                    Proximity<2> prox;
                    prox.index = Vec2u(vertex_index, index);
                    prox.value = Vec2f(1.0f, -1.0f);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, param.contact_ghat, offset, prop,
                        param.barrier, dt, count, param, stage, vertex_index);
                    return true;
                }
            }
        }
        return false;
    }
};

struct PointEdgeContactForceHessEmbed {

    unsigned vertex_index;
    const Kinematic &kinematic;
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
    const Vec<VertexProp> &prop;
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        Vec2u f = edge[index];
        if (f[0] != vertex_index && f[1] != vertex_index &&
            check_kinematic_pair(kinematic.edge[index],
                                 kinematic.vertex[vertex_index].active)) {
            const Vec3f &p = eval_x[vertex_index];
            const Vec3f &t0 = eval_x[f[0]];
            const Vec3f &t1 = eval_x[f[1]];
            Vec2f c = distance::point_edge_distance_coeff(p, t0, t1);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f e = c[0] * (p - t0) + c[1] * (p - t1);
                float offset =
                    make_vertex_offset(vertex_index, kinematic.vertex, param) +
                    make_edge_offset(f, kinematic.vertex, param);
                float ghat_sqr = sqr(param.contact_ghat + offset);
                float sqr_dist = e.squaredNorm();
                if (sqr_dist < ghat_sqr) {
                    unsigned count = 0;
                    if (neighbor.face.nnz) {
                        if (neighbor.face.count(index)) {
                            for (unsigned j = neighbor.face.offset[index];
                                 j < neighbor.face.offset[index + 1]; ++j) {
                                Vec3u f = face[neighbor.face.data[j]];
                                Vec3f c =
                                    distance::point_triangle_distance_coeff(
                                        p, eval_x[f[0]], eval_x[f[1]],
                                        eval_x[f[2]]);
                                if (c.maxCoeff() < 1.0f &&
                                    c.minCoeff() > 0.0f) {
                                    continue;
                                } else {
                                    ++count;
                                }
                            }
                        } else {
                            count = 1;
                        }
                    } else {
                        count = 1;
                    }
                    if (count) {
                        Proximity<3> prox;
                        prox.index = Vec3u(vertex_index, f[0], f[1]);
                        prox.value = Vec3f(1.0f, -c[0], -c[1]);
                        embed_contact_force_hess(
                            prox, vertex, eval_x, fixed_hess_in, fixed_out,
                            force, force_in, dyn_out, param.contact_ghat,
                            offset, prop, param.barrier, dt, count, param,
                            stage, vertex_index);
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
    const Kinematic &kinematic;
    const Vec<Vec3u> &face;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &prop;
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        Vec3u f = face[index];
        if (f[0] != vertex_index && f[1] != vertex_index &&
            f[2] != vertex_index &&
            check_kinematic_pair(kinematic.face[index],
                                 kinematic.vertex[vertex_index].active)) {
            const Vec3f &p = eval_x[vertex_index];
            const Vec3f &t0 = eval_x[f[0]];
            const Vec3f &t1 = eval_x[f[1]];
            const Vec3f &t2 = eval_x[f[2]];
            Vec3f c = distance::point_triangle_distance_coeff(p, t0, t1, t2);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f e = c[0] * (p - t0) + c[1] * (p - t1) + c[2] * (p - t2);
                float offset =
                    make_face_offset(param) +
                    make_vertex_offset(vertex_index, kinematic.vertex, param);
                float ghat_sqr = sqr(param.contact_ghat + offset);
                float sqr_dist = e.squaredNorm();
                if (sqr_dist < ghat_sqr) {
                    Proximity<4> prox;
                    prox.index = Vec4u(vertex_index, f[0], f[1], f[2]);
                    prox.value = Vec4f(1.0f, -c[0], -c[1], -c[2]);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, param.contact_ghat, offset, prop,
                        param.barrier, dt, 1, param, stage, vertex_index);
                    return true;
                }
            }
        }
        return false;
    }
};

struct EdgeEdgeContactForceHessEmbed {

    unsigned edge_index;
    const Kinematic &kinematic;
    const Vec<Vec2u> &edge;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &prop;
    int stage;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        const Vec2u &e0 = edge[edge_index];
        const Vec2u &e1 = edge[index];
        if (edge_index < index && intersect_free(e0[0], e0[1], e1[0], e1[1]) &&
            check_kinematic_pair(kinematic.edge[edge_index],
                                 kinematic.edge[index])) {
            Vec3f p0 = eval_x[e0[0]];
            Vec3f p1 = eval_x[e0[1]];
            Vec3f q0 = eval_x[e1[0]];
            Vec3f q1 = eval_x[e1[1]];
            float s(0.25f);
            Vec4f c = distance::edge_edge_distance_coeff(p0, p1, q0, q1);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f x0 = c[0] * p0 + c[1] * p1;
                Vec3f x1 = c[2] * q0 + c[3] * q1;
                Vec3f e = x0 - x1;
                float offset = make_edge_offset(e0, kinematic.vertex, param) +
                               make_edge_offset(e1, kinematic.vertex, param);
                float ghat_sqr = sqr(param.contact_ghat + offset);
                float sqr_dist = e.squaredNorm();
                if (sqr_dist < ghat_sqr) {
                    Proximity<4> prox;
                    prox.index = Vec4u(e0[0], e0[1], e1[0], e1[1]);
                    prox.value = Vec4f(c[0], c[1], -c[2], -c[3]);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, param.contact_ghat, offset, prop,
                        param.barrier, dt, 1, param, stage, edge_index);
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
    const Kinematic &kinematic;
    Vec<float> force;
    const Mat3x3f &local_hess;
    FixedCSRMat &dyn_out;
    const Vec<Vec3u> &collision_mesh_face;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &prop;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.vertex[vertex_index].active, true)) {
            const Vec3u &f = collision_mesh_face[index];
            Vec3f p = eval_x[vertex_index];
            Vec3f q = vertex[vertex_index] - p;
            Vec3f t0 = (collision_mesh_vertex[f[0]] - p);
            Vec3f t1 = (collision_mesh_vertex[f[1]] - p);
            Vec3f t2 = (collision_mesh_vertex[f[2]] - p);
            Vec3f zero = Vec3f::Zero();
            Vec3f c = distance::point_triangle_distance_coeff_unclassified(
                zero, t0, t1, t2);
            Vec3f y = c[0] * t0 + c[1] * t1 + c[2] * t2;
            Vec3f e = -y;
            Vec3f normal = e.normalized();
            float d = e.norm();
            if (d < param.contact_ghat) {
                float mass = prop[vertex_index].mass;
                Vec3f proj_x = y + normal * param.contact_ghat;
                float gap_squared = e.squaredNorm();
                float stiff_k =
                    normal.dot(local_hess * normal) + mass / gap_squared;
                Vec3f f = stiff_k *
                          push::gradient(-proj_x, normal, param.contact_ghat);
                Mat3x3f H = stiff_k *
                            push::hessian(-proj_x, normal, param.contact_ghat);
                Friction friction(f, -q, normal, param.friction,
                                  param.friction_eps);
                f += friction.gradient();
                H += friction.hessian();

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
    const Kinematic &kinematic;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    Vec<float> force;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &prop;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.face[index], true)) {
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
                distance::point_triangle_distance_coeff_unclassified(zero, t0,
                                                                     t1, t2);
            Vec3f p = c[0] * t0 + c[1] * t1 + c[2] * t2;
            Vec3f q = c[0] * s0 + c[1] * s1 + c[2] * s2;
            const Vec3f &e = p;
            Vec3f dx = p - q;
            Vec3f normal = e.normalized();
            float d = e.norm();
            if (d < param.contact_ghat) {
                Mat9x9f local_hess = Mat9x9f::Zero();
                float sqr_e = e.squaredNorm();
                for (unsigned ii = 0; ii < 3; ++ii) {
                    for (unsigned jj = 0; jj < 3; ++jj) {
                        local_hess.block<3, 3>(3 * ii, 3 * jj) =
                            fixed_hess_in(fc[ii], fc[jj]);
                    }
                    local_hess.block<3, 3>(3 * ii, 3 * ii) +=
                        (prop[fc[ii]].mass / sqr_e) * Mat3x3f::Identity();
                }
                Vec9f normal_ext;
                for (int j = 0; j < 9; ++j) {
                    normal_ext[j] = c[j / 3] * normal[j % 3];
                }
                float stiff_k = (local_hess * normal_ext).dot(normal_ext) /
                                normal_ext.squaredNorm();
                float area = c[0] * prop[fc[0]].area + c[1] * prop[fc[1]].area +
                             c[2] * prop[fc[2]].area;

                Vec3f proj_x = normal * param.contact_ghat;
                Vec3f f = stiff_k * push::gradient(p - proj_x, normal,
                                                   param.contact_ghat);
                Mat3x3f H = stiff_k * push::hessian(p - proj_x, normal,
                                                    param.contact_ghat);
                Friction friction(f, p - q, normal, param.friction,
                                  param.friction_eps);
                f += friction.gradient();
                H += friction.hessian();
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
    const Kinematic &kinematic;
    const Vec<Vec2u> &mesh_edge;
    Vec<float> &force;
    const Mat6x6f &local_hess;
    FixedCSRMat &dyn_out;
    const Vec<Vec2u> &collision_mesh_edge;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &prop;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.edge[i], true)) {
            const Vec2u &mesh_edge = this->mesh_edge[i];
            const Vec2u &coll_edge = collision_mesh_edge[index];
            Vec3f p0 = eval_x[mesh_edge[0]];
            Vec3f p1 = eval_x[mesh_edge[1]];
            Vec3f q0 = collision_mesh_vertex[coll_edge[0]];
            Vec3f q1 = collision_mesh_vertex[coll_edge[1]];
            Vec3f b0 = vertex[mesh_edge[0]];
            Vec3f b1 = vertex[mesh_edge[1]];
            Vec4f c =
                distance::edge_edge_distance_coeff_unclassified(p0, p1, q0, q1);
            Vec3f x = c[0] * p0 + c[1] * p1;
            Vec3f y = c[2] * q0 + c[3] * q1;
            Vec3f z = c[0] * b0 + c[1] * b1;
            Vec3f e = x - y;
            Vec3f dx = x - z;
            Vec3f normal = e.normalized();
            float d = e.norm();
            if (d < param.contact_ghat) {
                Vec3f proj_x = param.contact_ghat * normal;
                Vec6f normal_ext;
                for (int j = 0; j < 6; ++j) {
                    normal_ext[j] = c[j / 3] * normal[j % 3];
                }
                Mat6x6f mass_diag = Mat6x6f::Zero();
                float sqr_e = e.squaredNorm();
                mass_diag.block<3, 3>(0, 0) =
                    (prop[mesh_edge[0]].mass / sqr_e) * Mat3x3f::Identity();
                mass_diag.block<3, 3>(3, 3) =
                    (prop[mesh_edge[1]].mass / sqr_e) * Mat3x3f::Identity();
                float stiff_k =
                    ((local_hess + mass_diag) * normal_ext).dot(normal_ext) /
                    normal_ext.squaredNorm();
                float area = c[0] * prop[mesh_edge[0]].area +
                             c[1] * prop[mesh_edge[1]].area;
                Vec3f f = stiff_k * push::gradient(e - proj_x, normal,
                                                   param.contact_ghat);
                Mat3x3f H = stiff_k * push::hessian(e - proj_x, normal,
                                                    param.contact_ghat);

                Friction friction(f, x - z, normal, param.friction,
                                  param.friction_eps);
                f += friction.gradient();
                H += friction.hessian();

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
    const DataSet &data, const Vec<Vec3f> &eval_x, const Kinematic &kinematic,
    Vec<float> &force, const FixedCSRMat &fixed_hess_in, FixedCSRMat &fixed_out,
    const CollisionHessForceEmbedArgs &args, float dt, const ParamSet &param,
    unsigned i) {

    float ghat = param.constraint_ghat;
    float area = data.prop.vertex[i].area;
    float mass = data.prop.vertex[i].mass;
    unsigned num_contact = 0;

    Mat3x3f local_hess = fixed_hess_in(i, i);
    Mat3x3f H = Mat3x3f::Zero();
    Vec3f f = Vec3f::Zero();
    const Vec3f &x = eval_x[i];
    const Vec3f &dx = x - data.vertex.curr[i];

    if (kinematic.vertex[i].active) {
        const Vec3f &y = kinematic.vertex[i].position;
        Vec3f w = x - y;
        float d = w.norm();
        float gap = ghat - d;
        if (kinematic.vertex[i].kinematic) {
            gap = fmaxf(gap, param.constraint_tol * ghat);
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

                    Friction friction(f_push, dx, normal, param.friction,
                                      param.friction_eps);
                    f += friction.gradient();
                    H += friction.hessian();
                }
            }
        }

        for (unsigned j = 0; j < data.constraint.floor.size; ++j) {
            const Floor &floor = data.constraint.floor[j];
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
                Friction friction(f_push, dx, up, param.friction,
                                  param.friction_eps);
                f += friction.gradient();
                H += friction.hessian();
            }
        }
    }

    if (data.constraint.mesh.active) {
        CollisionMeshVertexFaceContactForceHessEmbed_M2C embed = {
            i,
            kinematic,
            force,
            local_hess,
            fixed_out,
            args.collision_mesh_face,
            args.collision_mesh_vertex,
            data.vertex.curr,
            eval_x,
            data.prop.vertex,
            dt,
            param};

        AABB pt_aabb = aabb::make(eval_x[i], 0.5f * param.contact_ghat);
        AABB_AABB_Tester<CollisionMeshVertexFaceContactForceHessEmbed_M2C> op(
            embed);
        num_contact += aabb::query(args.collision_mesh_face_bvh,
                                   args.collision_mesh_face_aabb, op, pt_aabb);
    }

    utility::atomic_embed_force<1>(Vec1u(i), f, force);
    utility::atomic_embed_hessian<1>(Vec1u(i), H, fixed_out);

    return num_contact;
}

unsigned embed_contact_force_hessian(
    const DataSet &data, const Vec<Vec3f> &eval_x, const Kinematic &kinematic,
    Vec<float> force, const FixedCSRMat &fixed_hess_in, FixedCSRMat &fixed_out,
    DynCSRMat &dyn_out, unsigned &max_nnz_row, float &dyn_consumed, float dt,
    const ParamSet &param) {

    // Name: Contact Matrix Assembly Time
    // Format: list[(vid_time,ms)]
    // Description:
    // Time spent in contact matrix assembly.
    SimpleLog logging("contact matrix assembly");

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned rod_count = data.rod_count;
    unsigned edge_count = data.mesh.mesh.edge.size;
    const BVH face_bvh = data.bvh.face;
    const BVH edge_bvh = data.bvh.edge;
    const BVH vertex_bvh = data.bvh.vertex;
    const Vec<AABB> face_aabb = storage::face_aabb;
    const Vec<AABB> edge_aabb = storage::edge_aabb;
    const Vec<AABB> vertex_aabb = storage::vertex_aabb;
    Vec<float> contact_force = storage::contact_force;
    Vec<unsigned> num_contact_vtf = storage::num_contact_vertex_face;
    Vec<unsigned> num_contact_ee = storage::num_contact_edge_edge;

    num_contact_vtf.clear(0);
    num_contact_ee.clear(0);
    contact_force.clear(0.0f);

    for (int stage = 0; stage < 2; ++stage) {
        if (stage == 0) {
            // Name: Dry Pass Time for Counting Matrix Nonzeros
            // Format: list[(vid_time,ms)]
            // Description:
            // Time for a dry pass to count matrix nonzeros.
            // This pass does not assemble the contact matrix.
            logging.push("dry pass");
            dyn_out.start_rebuild_buffer();
        } else {
            // Name: Fillin Pass Time for Assembling Contact Matrix
            // Format: list[(vid_time,ms)]
            // Description:
            // Time spent in fill-in pass for assembling the contact matrix.
            logging.push("fillin pass");
        }
        DISPATCH_START(surface_vert_count)
        [data, kinematic, eval_x, rod_count, contact_force, force,
         fixed_hess_in, fixed_out, dyn_out, face_bvh, face_aabb, edge_bvh,
         edge_aabb, vertex_bvh, vertex_aabb, num_contact_vtf, stage, dt,
         param] __device__(unsigned i) mutable {
            unsigned count(0);
            float ext_eps = 0.5f * param.contact_ghat +
                            make_vertex_offset(i, kinematic.vertex, param);
            AABB pt_aabb = aabb::make(eval_x[i], ext_eps);

            PointFaceContactForceHessEmbed embed_0 = {i,
                                                      kinematic,
                                                      data.mesh.mesh.face,
                                                      data.vertex.curr,
                                                      eval_x,
                                                      contact_force,
                                                      force,
                                                      fixed_hess_in,
                                                      fixed_out,
                                                      dyn_out,
                                                      data.prop.vertex,
                                                      stage,
                                                      dt,
                                                      param};

            AABB_AABB_Tester<PointFaceContactForceHessEmbed> op_0(embed_0);
            count += aabb::query(face_bvh, face_aabb, op_0, pt_aabb);

            PointEdgeContactForceHessEmbed embed_1 = {i,
                                                      kinematic,
                                                      data.mesh.mesh.edge,
                                                      data.mesh.mesh.face,
                                                      data.mesh.neighbor.edge,
                                                      data.vertex.curr,
                                                      eval_x,
                                                      contact_force,
                                                      force,
                                                      fixed_hess_in,
                                                      fixed_out,
                                                      dyn_out,
                                                      data.prop.vertex,
                                                      stage,
                                                      dt,
                                                      param};

            AABB_AABB_Tester<PointEdgeContactForceHessEmbed> op_1(embed_1);
            count += aabb::query(edge_bvh, edge_aabb, op_1, pt_aabb);

            PointPointContactForceHessEmbed embed_2 = {
                i,
                rod_count,
                kinematic,
                data.mesh.mesh.edge,
                data.mesh.mesh.face,
                data.mesh.neighbor.vertex,
                data.vertex.curr,
                eval_x,
                contact_force,
                force,
                fixed_hess_in,
                fixed_out,
                dyn_out,
                data.prop.vertex,
                stage,
                dt,
                param};
            AABB_AABB_Tester<PointPointContactForceHessEmbed> op_2(embed_2);
            count += aabb::query(vertex_bvh, vertex_aabb, op_2, pt_aabb);
            if (stage == 0) {
                num_contact_vtf[i] += count;
            }
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, kinematic, eval_x, contact_force, force, fixed_hess_in,
         fixed_out, dyn_out, edge_bvh, edge_aabb, num_contact_ee, stage, dt,
         param] __device__(unsigned i) mutable {
            Vec2u edge = data.mesh.mesh.edge[i];
            float ext_eps = 0.5f * param.contact_ghat +
                            make_edge_offset(edge, kinematic.vertex, param);
            AABB aabb = aabb::make(eval_x[edge[0]], eval_x[edge[1]], ext_eps);
            EdgeEdgeContactForceHessEmbed embed = {i,
                                                   kinematic,
                                                   data.mesh.mesh.edge,
                                                   data.vertex.curr,
                                                   eval_x,
                                                   contact_force,
                                                   force,
                                                   fixed_hess_in,
                                                   fixed_out,
                                                   dyn_out,
                                                   data.prop.vertex,
                                                   stage,
                                                   dt,
                                                   param};
            AABB_AABB_Tester<EdgeEdgeContactForceHessEmbed> op(embed);
            unsigned count = aabb::query(edge_bvh, edge_aabb, op, aabb);
            if (stage == 0) {
                num_contact_ee[i] += count;
            }
        } DISPATCH_END;
        logging.pop();

        if (stage == 0) {
            // Name: Time for Rebuilding Memory Layout for Contact Matrix
            // Format: list[(vid_time,ms)]
            // Map: contact_mat_rebuild
            // Description:
            // After the dry pass, the memory layout for the contact matrix is
            // re-computed so that the matrix can be assembled in the fill-in
            // pass.
            logging.push("rebuild");
            dyn_out.finish_rebuild_buffer(max_nnz_row, dyn_consumed);
            logging.pop();
        } else {
            // Name: Time for Filializing Contact Matrix
            // Format: list[(vid_time,ms)]
            // Map: contact_mat_finalize
            // Description:
            // After the fill-in pass, the contact matrix is compressed to
            // eliminate redundant entries.
            logging.push("finalize");
            dyn_out.finalize();
            logging.pop();
        }
    }

    DISPATCH_START(3 * surface_vert_count)
    [force, contact_force] __device__(unsigned i) mutable {
        force[i] += contact_force[i];
    } DISPATCH_END;

    return utility::sum_integer_array(num_contact_vtf, num_contact_vtf.size) +
           utility::sum_integer_array(num_contact_ee, num_contact_ee.size);
}

unsigned embed_constraint_force_hessian(
    const DataSet &data, const Vec<Vec3f> &eval_x, const Kinematic &kinematic,
    Vec<float> force, const FixedCSRMat &fixed_hess_in, FixedCSRMat &fixed_out,
    float dt, const ParamSet &param) {

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned edge_count = data.mesh.mesh.edge.size;

    const BVH &face_bvh = data.bvh.face;
    const Vec<AABB> &face_aabb = storage::face_aabb;
    Vec<unsigned> &num_contact_vtf = storage::num_contact_vertex_face;
    Vec<unsigned> &num_contact_ee = storage::num_contact_edge_edge;

    num_contact_vtf.clear(0);
    num_contact_ee.clear(0);

    CollisionHessForceEmbedArgs args = {
        data.constraint.mesh.vertex,      data.constraint.mesh.face,
        data.constraint.mesh.edge,        data.constraint.mesh.face_bvh,
        data.constraint.mesh.edge_bvh,    storage::collision_mesh_face_aabb,
        storage::collision_mesh_edge_aabb};

    DISPATCH_START(surface_vert_count)
    [data, eval_x, kinematic, force, fixed_hess_in, fixed_out, args, dt,
     num_contact_vtf, param] __device__(unsigned i) mutable {
        num_contact_vtf[i] += embed_vertex_constraint_force_hessian(
            data, eval_x, kinematic, force, fixed_hess_in, fixed_out, args, dt,
            param, i);
    } DISPATCH_END;

    if (data.constraint.mesh.active) {
        DISPATCH_START(data.constraint.mesh.vertex.size)
        [data, kinematic, eval_x, force, fixed_hess_in, fixed_out, args,
         num_contact_vtf, face_bvh, face_aabb, dt,
         param] __device__(unsigned i) mutable {
            float ext_eps = 0.5f * param.contact_ghat;
            CollisionMeshVertexFaceContactForceHessEmbed_C2M embed = {
                data,
                i,
                kinematic,
                fixed_hess_in,
                fixed_out,
                force,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                dt,
                param};
            AABB_AABB_Tester<CollisionMeshVertexFaceContactForceHessEmbed_C2M>
                op(embed);
            num_contact_vtf[i] += aabb::query(
                face_bvh, face_aabb, op,
                aabb::make(data.constraint.mesh.vertex[i], ext_eps));
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, kinematic, eval_x, force, fixed_hess_in, fixed_out, args,
         num_contact_ee, dt, param] __device__(unsigned i) mutable {
            const Vec2u &edge = data.mesh.mesh.edge[i];
            float ext_eps = 0.5f * param.contact_ghat +
                            make_edge_offset(edge, kinematic.vertex, param);
            Mat6x6f local_hess = Mat6x6f::Zero();
            for (unsigned ii = 0; ii < 2; ++ii) {
                for (unsigned jj = 0; jj < 2; ++jj) {
                    local_hess.block<3, 3>(3 * ii, 3 * jj) =
                        fixed_hess_in(edge[ii], edge[jj]);
                }
            }
            CollisionMeshEdgeEdgeContactForceHessEmbed embed = {
                i,
                kinematic,
                data.mesh.mesh.edge,
                force,
                local_hess,
                fixed_out,
                data.constraint.mesh.edge,
                data.constraint.mesh.vertex,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                dt,
                param};
            AABB aabb = aabb::make(eval_x[edge[0]], eval_x[edge[1]], ext_eps);
            AABB_AABB_Tester<CollisionMeshEdgeEdgeContactForceHessEmbed> op(
                embed);
            num_contact_ee[i] +=
                aabb::query(args.collision_mesh_edge_bvh,
                            args.collision_mesh_edge_aabb, op, aabb);
        } DISPATCH_END;
    }

    return utility::sum_integer_array(num_contact_vtf, num_contact_vtf.size) +
           utility::sum_integer_array(num_contact_ee, num_contact_ee.size);
}

struct CollisionMeshPointFaceCCD_M2C {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<Vec3u> &collision_mesh_face;
    const Vec<Vec3f> &collision_mesh_vertex;
    unsigned vertex_index;
    const Kinematic &kinematic;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.vertex[vertex_index].active, true)) {
            const Vec3u &f = collision_mesh_face[index];
            const Vec3f &t0 = collision_mesh_vertex[f[0]];
            const Vec3f &t1 = collision_mesh_vertex[f[1]];
            const Vec3f &t2 = collision_mesh_vertex[f[2]];
            const Vec3f &p0 = x0[vertex_index];
            const Vec3f &p1 = x1[vertex_index];
            float result = accd::point_triangle_ccd(p0, p1, t0, t1, t2, t0, t1,
                                                    t2, 0.0f, param);
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

struct CollisionMeshPointFaceCCD_C2M {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<Vec3u> &collision_mesh_face;
    const Vec<Vec3f> &collision_mesh_vertex;
    unsigned vertex_index;
    const Kinematic &kinematic;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.face[index], true)) {
            const Vec3u &f = face[index];
            const Vec3f &t00 = x0[f[0]];
            const Vec3f &t01 = x0[f[1]];
            const Vec3f &t02 = x0[f[2]];
            const Vec3f &t10 = x1[f[0]];
            const Vec3f &t11 = x1[f[1]];
            const Vec3f &t12 = x1[f[2]];
            const Vec3f &p = collision_mesh_vertex[vertex_index];
            float result = accd::point_triangle_ccd(p, p, t00, t01, t02, t10,
                                                    t11, t12, 0.0f, param);
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
    unsigned edge_index;
    const Kinematic &kinematic;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.edge[edge_index], true)) {
            const Vec2u &e0 = edge[edge_index];
            const Vec2u &e1 = collision_mesh_edge[index];
            const Vec3f &p00 = x0[e0[0]];
            const Vec3f &p01 = x0[e0[1]];
            const Vec3f &p10 = x1[e0[0]];
            const Vec3f &p11 = x1[e0[1]];
            const Vec3f &q0 = collision_mesh_vertex[e1[0]];
            const Vec3f &q1 = collision_mesh_vertex[e1[1]];
            float result = accd::edge_edge_ccd(p00, p01, q0, q1, p10, p11, q0,
                                               q1, 0.0f, param);
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
    unsigned vertex_index;
    const Kinematic &kinematic;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        const Vec3u &f = face[index];
        if (f[0] != vertex_index && f[1] != vertex_index &&
            f[2] != vertex_index &&
            check_kinematic_pair(kinematic.vertex[vertex_index].active,
                                 kinematic.face[index])) {
            float offset =
                make_face_offset(param) +
                make_vertex_offset(vertex_index, kinematic.vertex, param);
            const Vec3f &p0 = x0[vertex_index];
            const Vec3f &p1 = x1[vertex_index];
            const Vec3f &t00 = x0[f[0]];
            const Vec3f &t01 = x0[f[1]];
            const Vec3f &t02 = x0[f[2]];
            const Vec3f &t10 = x1[f[0]];
            const Vec3f &t11 = x1[f[1]];
            const Vec3f &t12 = x1[f[2]];
            float result = accd::point_triangle_ccd(p0, p1, t00, t01, t02, t10,
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

struct EdgeEdgeCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec2u> &edge;
    unsigned edge_index;
    const Kinematic &kinematic;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        const Vec2u &e0 = edge[edge_index];
        const Vec2u &e1 = edge[index];
        if (edge_index < index &&
            check_kinematic_pair(kinematic.edge[edge_index],
                                 kinematic.edge[index])) {
            if (intersect_free(e0[0], e0[1], e1[0], e1[1])) {
                float offset = make_edge_offset(e0, kinematic.vertex, param) +
                               make_edge_offset(e1, kinematic.vertex, param);
                const Vec3f &p00 = x0[e0[0]];
                const Vec3f &p01 = x0[e0[1]];
                const Vec3f &q00 = x0[e1[0]];
                const Vec3f &q01 = x0[e1[1]];
                const Vec3f &p10 = x1[e0[0]];
                const Vec3f &p11 = x1[e0[1]];
                const Vec3f &q10 = x1[e1[0]];
                const Vec3f &q11 = x1[e1[1]];
                float result = accd::edge_edge_ccd(p00, p01, q00, q01, p10, p11,
                                                   q10, q11, offset, param);
                if (result < param.line_search_max_t) {
                    toi = fminf(toi, result);
                    assert(toi > 0.0f);
                    return true;
                }
            }
        }
        return false;
    }
};

__device__ void
vertex_constraint_line_search(const DataSet &data, const Kinematic &kinematic,
                              const Vec<Vec3f> &y0, const Vec<Vec3f> &y1,
                              Vec<float> toi_vert, ParamSet param, unsigned i) {
    const Vec3f x1 = param.line_search_max_t * (y1[i] - y0[i]) + y0[i];
    const Vec3f &x0 = y0[i];
    if (kinematic.vertex[i].active) {
        if (kinematic.vertex[i].kinematic == false) {
            const Vec3f &position = kinematic.vertex[i].position;
            float r0 = (x0 - position).norm();
            float r1 = (x1 - position).norm();
            assert(r0 < param.constraint_ghat);
            float r = param.constraint_ghat;
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

float line_search(const DataSet &data, const Kinematic &kinematic,
                  const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                  const ParamSet &param) {

    const MeshInfo &mesh = data.mesh;
    const BVHSet &bvhset = data.bvh;

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned edge_count = mesh.mesh.edge.size;

    const BVH &face_bvh = bvhset.face;
    const BVH &edge_bvh = bvhset.edge;
    const BVH &collision_mesh_face_bvh = data.constraint.mesh.face_bvh;
    const BVH &collision_mesh_edge_bvh = data.constraint.mesh.edge_bvh;
    const Vec<Vec2u> &collision_mesh_edge = data.constraint.mesh.edge;
    const Vec<Vec3u> &collision_mesh_face = data.constraint.mesh.face;
    const Vec<Vec3f> &collision_mesh_vertex = data.constraint.mesh.vertex;
    const Vec<AABB> &face_aabb = storage::face_aabb;
    const Vec<AABB> &edge_aabb = storage::edge_aabb;
    const Vec<AABB> &collision_mesh_face_aabb =
        storage::collision_mesh_face_aabb;
    const Vec<AABB> &collision_mesh_edge_aabb =
        storage::collision_mesh_edge_aabb;

    Vec<char> &intersection_flag = storage::intersection_flag;
    Vec<unsigned> &num_contact_vtf = storage::num_contact_vertex_face;
    Vec<unsigned> &num_contact_ee = storage::num_contact_edge_edge;
    Vec<float> &toi_vtf = storage::vertex_face_toi;
    Vec<float> &toi_ee = storage::edge_edge_toi;

    intersection_flag.clear(0);
    toi_vtf.clear(param.line_search_max_t);
    toi_ee.clear(param.line_search_max_t);

    DISPATCH_START(surface_vert_count)
    [data, kinematic, mesh, x0, x1, face_bvh, face_aabb, num_contact_vtf,
     toi_vtf, param] __device__(unsigned i) mutable {
        float ext_eps = 0.5f * param.contact_ghat +
                        make_vertex_offset(i, kinematic.vertex, param);
        float toi = param.line_search_max_t;
        PointFaceCCD ccd = {x0, x1, mesh.mesh.face, i, kinematic, toi, param};
        AABB_AABB_Tester<PointFaceCCD> op(ccd);
        AABB aabb = aabb::make(x0[i], x1[i], ext_eps);
        num_contact_vtf[i] = aabb::query(face_bvh, face_aabb, op, aabb);
        toi_vtf[i] = fmin(toi_vtf[i], toi);
        vertex_constraint_line_search(data, kinematic, x0, x1, toi_vtf, param,
                                      i);
    } DISPATCH_END;

    if (data.constraint.mesh.active) {
        DISPATCH_START(surface_vert_count)
        [mesh, kinematic, x0, x1, collision_mesh_face, collision_mesh_face_bvh,
         collision_mesh_face_aabb, collision_mesh_vertex, num_contact_vtf,
         toi_vtf, param] __device__(unsigned i) mutable {
            float ext_eps = 0.5f * param.contact_ghat +
                            make_vertex_offset(i, kinematic.vertex, param);
            float toi = param.line_search_max_t;
            CollisionMeshPointFaceCCD_M2C ccd = {x0,
                                                 x1,
                                                 mesh.mesh.face,
                                                 collision_mesh_face,
                                                 collision_mesh_vertex,
                                                 i,
                                                 kinematic,
                                                 toi,
                                                 param};
            AABB_AABB_Tester<CollisionMeshPointFaceCCD_M2C> op(ccd);
            AABB aabb = aabb::make(x0[i], x1[i], ext_eps);
            num_contact_vtf[i] = aabb::query(
                collision_mesh_face_bvh, collision_mesh_face_aabb, op, aabb);
            toi_vtf[i] = fmin(toi_vtf[i], toi);
        } DISPATCH_END;

        unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
        DISPATCH_START(collision_mesh_vert_count)
        [mesh, kinematic, x0, x1, collision_mesh_face, collision_mesh_vertex,
         num_contact_vtf, toi_vtf, face_bvh, face_aabb,
         param] __device__(unsigned i) mutable {
            float toi = param.line_search_max_t;
            CollisionMeshPointFaceCCD_C2M ccd = {x0,
                                                 x1,
                                                 mesh.mesh.face,
                                                 collision_mesh_face,
                                                 collision_mesh_vertex,
                                                 i,
                                                 kinematic,
                                                 toi,
                                                 param};
            AABB_AABB_Tester<CollisionMeshPointFaceCCD_C2M> op(ccd);
            Vec3f q = collision_mesh_vertex[i];
            num_contact_vtf[i] =
                aabb::query(face_bvh, face_aabb, op,
                            aabb::make(q, 0.5f * param.contact_ghat));
            toi_vtf[i] = fmin(toi_vtf[i], toi);
        } DISPATCH_END;
    }

    DISPATCH_START(edge_count)
    [mesh, kinematic, x0, x1, edge_bvh, edge_aabb, num_contact_ee, toi_ee,
     param] __device__(unsigned i) mutable {
        Vec2u edge = mesh.mesh.edge[i];
        float ext_eps = 0.5f * param.contact_ghat +
                        make_edge_offset(edge, kinematic.vertex, param);
        AABB aabb0 = aabb::make(x0[edge[0]], x0[edge[1]], ext_eps);
        AABB aabb1 = aabb::make(x1[edge[0]], x1[edge[1]], ext_eps);
        AABB aabb = aabb::join(aabb0, aabb1);
        float toi = param.line_search_max_t;
        EdgeEdgeCCD ccd = {x0, x1, mesh.mesh.edge, i, kinematic, toi, param};
        AABB_AABB_Tester<EdgeEdgeCCD> op(ccd);
        num_contact_ee[i] = aabb::query(edge_bvh, edge_aabb, op, aabb);
        toi_ee[i] = fmin(toi, toi_ee[i]);
    } DISPATCH_END;

    if (data.constraint.mesh.active) {
        DISPATCH_START(edge_count)
        [mesh, kinematic, x0, x1, collision_mesh_edge_bvh, collision_mesh_edge,
         collision_mesh_vertex, collision_mesh_edge_aabb, num_contact_ee,
         toi_ee, param] __device__(unsigned i) mutable {
            Vec2u edge = mesh.mesh.edge[i];
            float ext_eps = 0.5f * param.contact_ghat +
                            make_edge_offset(edge, kinematic.vertex, param);
            AABB aabb0 = aabb::make(x0[edge[0]], x0[edge[1]], ext_eps);
            AABB aabb1 = aabb::make(x1[edge[0]], x1[edge[1]], ext_eps);
            AABB aabb = aabb::join(aabb0, aabb1);
            float toi = param.line_search_max_t;
            CollisionMeshEdgeEdgeCCD ccd = {x0,
                                            x1,
                                            mesh.mesh.edge,
                                            collision_mesh_edge,
                                            collision_mesh_vertex,
                                            i,
                                            kinematic,
                                            toi,
                                            param};

            AABB_AABB_Tester<CollisionMeshEdgeEdgeCCD> op(ccd);
            num_contact_ee[i] = aabb::query(collision_mesh_edge_bvh,
                                            collision_mesh_edge_aabb, op, aabb);
            toi_ee[i] = fmin(toi, toi_ee[i]);
        } DISPATCH_END;
    }

    float toi = fminf(
        utility::min_array(toi_vtf.data, toi_vtf.size, param.line_search_max_t),
        utility::min_array(toi_ee.data, toi_ee.size, param.line_search_max_t));
    return toi / param.line_search_max_t;
}

__device__ bool edge_triangle_intersect(const Vec3f &_e0, const Vec3f &_e1,
                                        const Vec3f &_x0, const Vec3f &_x1,
                                        const Vec3f &_x2) {
    Vec3f n = (_x1 - _x0).cross(_x2 - _x0);
    float s1 = (_e0 - _x0).dot(n);
    float s2 = (_e1 - _x0).dot(n);
    if (s1 * s2 < 0.0f) {
        Vec3f r = (_e1 - _e0) * s1 / (s1 - s2) + _e0;
        Vec3f c = distance::point_triangle_distance_coeff(r, _x0, _x1, _x2);
        if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
            return true;
        }
    }
    return false;
}

class FaceEdgeIntersectTester {
  public:
    __device__
    FaceEdgeIntersectTester(const Vec<Vec3f> &vertex, const Vec<Vec3u> &face,
                            const Vec<Vec2u> &edge, const Kinematic &kinematic,
                            unsigned edge_index)
        : vertex(vertex), face(face), edge(edge), kinematic(kinematic),
          edge_index(edge_index) {}
    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.edge[edge_index],
                                 kinematic.face[index])) {
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
    const Vec<Vec3f> &vertex;
    const Vec<Vec3u> &face;
    const Vec<Vec2u> &edge;
    const Kinematic &kinematic;
    unsigned edge_index;
};

class CollisionMeshFaceEdgeIntersectTester {
  public:
    __device__ CollisionMeshFaceEdgeIntersectTester(const Vec<Vec3f> &vertex,
                                                    const Vec<Vec3u> &face,
                                                    const Kinematic &kinematic,
                                                    const Vec3f &y0,
                                                    const Vec3f &y1)
        : vertex(vertex), face(face), kinematic(kinematic), y0(y0), y1(y1) {}
    __device__ bool operator()(unsigned index) {
        if (check_kinematic_pair(kinematic.face[index], true)) {
            Vec3u f = face[index];
            const Vec3f &x0 = vertex[f[0]];
            const Vec3f &x1 = vertex[f[1]];
            const Vec3f &x2 = vertex[f[2]];
            if (edge_triangle_intersect(y0, y1, x0, x1, x2)) {
                return true;
            }
        }
        return false;
    }
    const Vec<Vec3f> &vertex;
    const Vec<Vec3u> &face;
    const Kinematic &kinematic;
    Vec3f y0, y1;
};

bool check_intersection(const DataSet &data, const Kinematic &kinematic,
                        const Vec<Vec3f> &vertex) {

    unsigned edge_count = data.mesh.mesh.edge.size;
    const BVH &face_bvh = data.bvh.face;
    const BVH &collision_mesh_face_bvh = data.constraint.mesh.face_bvh;
    const MeshInfo &mesh = data.mesh;
    const Vec<Vec3u> &collision_mesh_face = data.constraint.mesh.face;
    const Vec<AABB> &face_aabb = storage::face_aabb;
    const Vec<AABB> &collision_mesh_face_aabb =
        storage::collision_mesh_face_aabb;
    Vec<char> &intersection_flag = storage::intersection_flag;
    intersection_flag.clear(0);
    const Vec<Vec3f> &collision_mesh_vertex = data.constraint.mesh.vertex;
    bool collision_mesh_active = data.constraint.mesh.active;

    DISPATCH_START(edge_count)
    [mesh, kinematic, vertex, collision_mesh_vertex, face_bvh, face_aabb,
     collision_mesh_face_bvh, collision_mesh_face_aabb, collision_mesh_face,
     intersection_flag, collision_mesh_active] __device__(unsigned i) mutable {
        const Vec2u &edge = mesh.mesh.edge[i];
        Vec3f y0 = vertex[edge[0]];
        Vec3f y1 = vertex[edge[1]];
        AABB aabb = aabb::make(y0, y1, 0.0f);
        FaceEdgeIntersectTester tester_0(vertex, mesh.mesh.face, mesh.mesh.edge,
                                         kinematic, i);
        AABB_AABB_Tester<FaceEdgeIntersectTester> op_0(tester_0);
        if (aabb::query(face_bvh, face_aabb, op_0, aabb)) {
            intersection_flag[i] = 1;
        }
        if (collision_mesh_active) {
            CollisionMeshFaceEdgeIntersectTester tester_1(
                collision_mesh_vertex, collision_mesh_face, kinematic, y0, y1);
            AABB_AABB_Tester<CollisionMeshFaceEdgeIntersectTester> op_1(
                tester_1);
            if (aabb::query(collision_mesh_face_bvh, collision_mesh_face_aabb,
                            op_1, aabb)) {
                intersection_flag[i] = 1;
            }
        }
    } DISPATCH_END;

    return utility::sum_integer_array(intersection_flag, edge_count) == 0;
}

} // namespace contact

#endif
