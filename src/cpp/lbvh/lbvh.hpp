// File: lbvh.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef LBVH_HPP
#define LBVH_HPP

#include "../data.hpp"

namespace lbvh {

// Initialize persistent GPU buffers for LBVH construction
void initialize(unsigned max_faces, unsigned max_edges, unsigned max_vertices);

// Build face BVH with swept AABBs covering motion from x0 to x0+extrapolate*(x1-x0)
void build_face_bvh(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                    float extrapolate, const Vec<Vec3u> &face, BVH &bvh,
                    Vec<AABB> &aabb, const Vec<FaceProp> &prop,
                    const Vec<FaceParam> &params);

// Build edge BVH with swept AABBs covering motion from x0 to x0+extrapolate*(x1-x0)
void build_edge_bvh(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                    float extrapolate, const Vec<Vec2u> &edge, BVH &bvh,
                    Vec<AABB> &aabb, const Vec<EdgeProp> &prop,
                    const Vec<EdgeParam> &params);

// Build vertex BVH with swept AABBs covering motion from x0 to x0+extrapolate*(x1-x0)
void build_vertex_bvh(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                      float extrapolate, BVH &bvh, Vec<AABB> &aabb,
                      unsigned surface_vert_count, const Vec<VertexProp> &prop,
                      const Vec<VertexParam> &params);

// Update AABBs only (reuse existing tree structure) - faster than full rebuild
void update_face_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                      float extrapolate, const Vec<Vec3u> &face,
                      const BVH &bvh, Vec<AABB> &aabb, const Vec<FaceProp> &prop,
                      const Vec<FaceParam> &params);

void update_edge_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                      float extrapolate, const Vec<Vec2u> &edge,
                      const BVH &bvh, Vec<AABB> &aabb, const Vec<EdgeProp> &prop,
                      const Vec<EdgeParam> &params);

void update_vertex_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                        float extrapolate, const BVH &bvh, Vec<AABB> &aabb,
                        unsigned surface_vert_count, const Vec<VertexProp> &prop,
                        const Vec<VertexParam> &params);

// Build collision mesh BVHs (static obstacles)
void build_collision_mesh_bvh(const DataSet &data, const ParamSet &param);

} // namespace lbvh

#endif // LBVH_HPP
