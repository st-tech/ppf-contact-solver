// File: bvh_storage.hpp
// BVH storage managed entirely on GPU side
// License: Apache v2.0

#ifndef BVH_STORAGE_HPP
#define BVH_STORAGE_HPP

#include "../data.hpp"

namespace bvh_storage {

// Get references to the BVH structures (managed by LBVH module)
BVHSet &get_bvh();
BVHSet &get_collision_mesh_bvh();

} // namespace bvh_storage

#endif
