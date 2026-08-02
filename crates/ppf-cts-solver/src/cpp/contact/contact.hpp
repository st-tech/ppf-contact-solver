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

unsigned
embed_contact_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                            Vec<float> force, const FixedCSRMat &fixed_hess_in,
                            FixedCSRMat &fixed_out, DynCSRMat &hess_out,
                            unsigned &max_nnz_row, float &dyn_consumed,
                            float dt, const ParamSet &param);

unsigned embed_constraint_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> force,
                                        const FixedCSRMat &fixed_hess_in,
                                        FixedCSRMat &fixed_hess_out, float dt,
                                        const ParamSet &param);

// `pin_infeasible` is a device scalar the caller clears to UINT_MAX. If a
// fix-pinned (prescribed) vertex's swept path crosses an analytic collider it
// cannot yield to, the smallest such vertex index is written there. The step
// must then fail: the prescription itself is unsatisfiable.
float line_search(const DataSet &data, const Vec<Vec3f> &x0,
                  const Vec<Vec3f> &x1, const ParamSet &param,
                  unsigned *pin_infeasible);

// Clear / read the "contact starts overlapping" device flag that the CCD sets
// when a contact pair begins the step inside the contact offset (two surfaces
// already touching / interpenetrating). The caller clears it before a line
// search and reads it after; a detected overlap fails the step with a
// structured OverlappingStart crash instead of a raw device assert.
void clear_ccd_overlap();
bool ccd_overlap_detected();
// Fills a representative overlapping vertex pair and its kind (0 = vertex-face,
// 1 = edge-edge, 2 = point-point among dynamic vertices; 3 = vertex-face,
// 4 = face-vertex, 5 = edge-edge against the static collision mesh, where v0
// is a dynamic vertex and v1 a collision-mesh vertex); UINT_MAX if none was
// recorded. d2 and offset are a flagged pair's squared start distance and
// contact offset in the CCD's internally rescaled units (-1 if unset); their
// ratio tells how deep the overlap was, and d2 == 0.0 exactly means the pair
// evaluates as touching at the resolution of that frame.
void ccd_overlap_info(unsigned &v0, unsigned &v1, unsigned &kind, float &d2,
                      float &offset);

bool check_intersection(const DataSet &data, const Vec<Vec3f> &vertex,
                        const ParamSet &param);

// AABB storage accessors for GPU BVH construction
Vec<AABB> &get_face_aabb();
Vec<AABB> &get_edge_aabb();
Vec<AABB> &get_vertex_aabb();
Vec<AABB> &get_collision_mesh_face_aabb();
Vec<AABB> &get_collision_mesh_edge_aabb();

// Intersection record accessors
unsigned get_intersection_count();
const IntersectionRecord *get_intersection_records();

// Collision active flag pointers (null if collision windows not initialized)
const bool *get_vert_collision_active();
const bool *get_edge_collision_active();
const bool *get_face_collision_active();

} // namespace contact

#endif
