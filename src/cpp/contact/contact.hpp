// File: contact.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CONTACT_DEF_HPP
#define CONTACT_DEF_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"

namespace contact {

void initialize(const DataSet &data, const ParamSet &param);
void resize_aabb(const BVHSet &bvh);
void update_aabb(const DataSet &host_data, const DataSet &dev_data,
                 const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                 const ParamSet &param);

void update_collision_mesh_aabb(const DataSet &host_data,
                                const DataSet &dev_data, const ParamSet &param);

unsigned embed_contact_force_hessian(const DataSet &data,
                                     const Vec<Vec3f> &eval_x, Vec<float> force,
                                     const FixedCSRMat &fixed_hess_in,
                                     FixedCSRMat &fixed_out,
                                     DynCSRMat &hess_out, unsigned &max_nnz_row,
                                     float &dyn_consumed, float dt,
                                     const ParamSet &param);

unsigned embed_constraint_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> force,
                                        const FixedCSRMat &fixed_hess_in,
                                        FixedCSRMat &fixed_hess_out, float dt,
                                        const ParamSet &param);

float line_search(const DataSet &data, const Vec<Vec3f> &x0,
                  const Vec<Vec3f> &x1, const ParamSet &param);

bool check_intersection(const DataSet &data, const Vec<Vec3f> &vertex,
                        const ParamSet &param);

} // namespace contact

#endif
