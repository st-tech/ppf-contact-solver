// File: lbvh.cu
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../buffer/buffer.hpp"
#include "../contact/contact.hpp"
#include "../kernels/radix_sort.hpp"
#include "../main/cuda_utils.hpp"
#include "../utility/dispatcher.hpp"
#include "bvh_storage.hpp"
#include "lbvh.hpp"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <vector>

namespace lbvh {

// Persistent GPU buffers for LBVH construction
namespace storage {
Vec<unsigned> morton_codes;
Vec<unsigned> sorted_indices;
Vec<unsigned> temp_morton;
Vec<unsigned> temp_indices;
Vec<unsigned> histogram_buffer;
Vec<float> centroids_x, centroids_y, centroids_z;
Vec<float> scene_bounds; // [min_x, min_y, min_z, max_x, max_y, max_z]
unsigned max_primitives;

// Tree building temporaries
Vec<unsigned> parent;
Vec<unsigned> depth;
Vec<unsigned> level_counts;
Vec<unsigned> level_offsets;
Vec<unsigned> level_positions;
Vec<unsigned> max_depth_storage; // Single element for max depth reduction
unsigned max_nodes;
constexpr unsigned MAX_LEVELS = 64;

// BVH storage - managed entirely on GPU
BVHSet main_bvh;
BVHSet collision_mesh_bvh;
} // namespace storage

void initialize(unsigned max_faces, unsigned max_edges, unsigned max_vertices) {
    unsigned max_n = max_faces;
    if (max_edges > max_n) {
        max_n = max_edges;
    }
    if (max_vertices > max_n) {
        max_n = max_vertices;
    }

    storage::max_primitives = max_n;
    unsigned max_nodes = max_n > 0 ? 2 * max_n - 1 : 0;
    storage::max_nodes = max_nodes;

    // Morton codes and indices
    storage::morton_codes = Vec<unsigned>::alloc(max_n);
    storage::sorted_indices = Vec<unsigned>::alloc(max_n);
    storage::temp_morton = Vec<unsigned>::alloc(max_n);
    storage::temp_indices = Vec<unsigned>::alloc(max_n);

    // Histogram for radix sort
    unsigned num_blocks =
        (max_n + kernels::SORT_BLOCK_SIZE - 1) / kernels::SORT_BLOCK_SIZE;
    unsigned hist_size = kernels::RADIX_SIZE * num_blocks;
    storage::histogram_buffer = Vec<unsigned>::alloc(hist_size);

    // Centroids and scene bounds
    storage::centroids_x = Vec<float>::alloc(max_n);
    storage::centroids_y = Vec<float>::alloc(max_n);
    storage::centroids_z = Vec<float>::alloc(max_n);
    storage::scene_bounds = Vec<float>::alloc(6);

    // Tree building temporaries
    storage::parent = Vec<unsigned>::alloc(max_nodes);
    storage::depth = Vec<unsigned>::alloc(max_nodes);
    storage::level_counts = Vec<unsigned>::alloc(storage::MAX_LEVELS);
    storage::level_offsets = Vec<unsigned>::alloc(storage::MAX_LEVELS + 1);
    storage::level_positions = Vec<unsigned>::alloc(storage::MAX_LEVELS);
    storage::max_depth_storage = Vec<unsigned>::alloc(1);

    // Pre-allocate BVH structures (reserve sets size=0, built later)
    storage::main_bvh.face.node = Vec<Vec2u>::reserve(max_nodes);
    storage::main_bvh.face.level =
        VecVec<unsigned>::alloc(storage::MAX_LEVELS, max_nodes);
    storage::main_bvh.edge.node = Vec<Vec2u>::reserve(max_nodes);
    storage::main_bvh.edge.level =
        VecVec<unsigned>::alloc(storage::MAX_LEVELS, max_nodes);
    storage::main_bvh.vertex.node = Vec<Vec2u>::reserve(max_nodes);
    storage::main_bvh.vertex.level =
        VecVec<unsigned>::alloc(storage::MAX_LEVELS, max_nodes);

    storage::collision_mesh_bvh.face.node = Vec<Vec2u>::reserve(max_nodes);
    storage::collision_mesh_bvh.face.level =
        VecVec<unsigned>::alloc(storage::MAX_LEVELS, max_nodes);
    storage::collision_mesh_bvh.edge.node = Vec<Vec2u>::reserve(max_nodes);
    storage::collision_mesh_bvh.edge.level =
        VecVec<unsigned>::alloc(storage::MAX_LEVELS, max_nodes);
}

//==============================================================================
// Morton code computation
//==============================================================================

__device__ unsigned expand_bits(unsigned v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

__device__ unsigned morton_code_3d(unsigned x, unsigned y, unsigned z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

__global__ void compute_face_centroids_kernel(const Vec3f *vertex,
                                              const Vec3u *face, unsigned n,
                                              float *cx, float *cy, float *cz) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    Vec3u f = face[i];
    Vec3f v0 = vertex[f[0]];
    Vec3f v1 = vertex[f[1]];
    Vec3f v2 = vertex[f[2]];

    cx[i] = (float(v0[0]) + float(v1[0]) + float(v2[0])) / 3.0f;
    cy[i] = (float(v0[1]) + float(v1[1]) + float(v2[1])) / 3.0f;
    cz[i] = (float(v0[2]) + float(v1[2]) + float(v2[2])) / 3.0f;
}

__global__ void compute_edge_centroids_kernel(const Vec3f *vertex,
                                              const Vec2u *edge, unsigned n,
                                              float *cx, float *cy, float *cz) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    Vec2u e = edge[i];
    Vec3f v0 = vertex[e[0]];
    Vec3f v1 = vertex[e[1]];

    cx[i] = (float(v0[0]) + float(v1[0])) / 2.0f;
    cy[i] = (float(v0[1]) + float(v1[1])) / 2.0f;
    cz[i] = (float(v0[2]) + float(v1[2])) / 2.0f;
}

__global__ void compute_vertex_centroids_kernel(const Vec3f *vertex,
                                                unsigned n, float *cx,
                                                float *cy, float *cz) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    Vec3f v = vertex[i];
    cx[i] = float(v[0]);
    cy[i] = float(v[1]);
    cz[i] = float(v[2]);
}

// Atomic min for floats using CAS (works correctly for all float values)
__device__ void atomicMinFloat(float *addr, float value) {
    int *addr_as_int = (int *)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_val = __int_as_float(expected);
        if (value >= old_val) {
            return; // Current value is already smaller
        }
        old = atomicCAS(addr_as_int, expected, __float_as_int(value));
    } while (old != expected);
}

// Atomic max for floats using CAS (works correctly for all float values)
__device__ void atomicMaxFloat(float *addr, float value) {
    int *addr_as_int = (int *)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_val = __int_as_float(expected);
        if (value <= old_val) {
            return; // Current value is already larger
        }
        old = atomicCAS(addr_as_int, expected, __float_as_int(value));
    } while (old != expected);
}

__global__ void compute_scene_bounds_kernel(const float *cx, const float *cy,
                                            const float *cz, unsigned n,
                                            float *bounds) {
    __shared__ float shared_min[3][256];
    __shared__ float shared_max[3][256];

    unsigned tid = threadIdx.x;
    unsigned gid = blockIdx.x * blockDim.x + tid;

    // Initialize with extreme values
    float local_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float local_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    if (gid < n) {
        local_min[0] = local_max[0] = cx[gid];
        local_min[1] = local_max[1] = cy[gid];
        local_min[2] = local_max[2] = cz[gid];
    }

    shared_min[0][tid] = local_min[0];
    shared_min[1][tid] = local_min[1];
    shared_min[2][tid] = local_min[2];
    shared_max[0][tid] = local_max[0];
    shared_max[1][tid] = local_max[1];
    shared_max[2][tid] = local_max[2];
    __syncthreads();

    // Reduction within block
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int d = 0; d < 3; ++d) {
                shared_min[d][tid] =
                    fminf(shared_min[d][tid], shared_min[d][tid + s]);
                shared_max[d][tid] =
                    fmaxf(shared_max[d][tid], shared_max[d][tid + s]);
            }
        }
        __syncthreads();
    }

    // First thread writes block result atomically
    if (tid == 0) {
        for (int d = 0; d < 3; ++d) {
            atomicMinFloat(&bounds[d], shared_min[d][0]);
            atomicMaxFloat(&bounds[d + 3], shared_max[d][0]);
        }
    }
}

__global__ void init_bounds_kernel(float *bounds) {
    bounds[0] = bounds[1] = bounds[2] = FLT_MAX;
    bounds[3] = bounds[4] = bounds[5] = -FLT_MAX;
}

__global__ void compute_morton_codes_kernel(const float *cx, const float *cy,
                                            const float *cz, unsigned n,
                                            const float *bounds,
                                            unsigned *morton_codes,
                                            unsigned *indices) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float min_x = bounds[0], min_y = bounds[1], min_z = bounds[2];
    float max_x = bounds[3], max_y = bounds[4], max_z = bounds[5];

    float scale_x = (max_x - min_x) > 1e-10f ? 1023.0f / (max_x - min_x) : 0.0f;
    float scale_y = (max_y - min_y) > 1e-10f ? 1023.0f / (max_y - min_y) : 0.0f;
    float scale_z = (max_z - min_z) > 1e-10f ? 1023.0f / (max_z - min_z) : 0.0f;

    unsigned ix =
        (unsigned)fminf(1023.0f, fmaxf(0.0f, (cx[i] - min_x) * scale_x));
    unsigned iy =
        (unsigned)fminf(1023.0f, fmaxf(0.0f, (cy[i] - min_y) * scale_y));
    unsigned iz =
        (unsigned)fminf(1023.0f, fmaxf(0.0f, (cz[i] - min_z) * scale_z));

    morton_codes[i] = morton_code_3d(ix, iy, iz);
    indices[i] = i;
}

//==============================================================================
// Tree building (GPU-based parallel construction)
//==============================================================================

// Count leading zeros for 32-bit unsigned
__device__ int clz(unsigned x) { return x == 0 ? 32 : __clz(x); }

// Compute longest common prefix length between Morton codes at indices i and j
__device__ int longest_common_prefix(const unsigned *morton_codes, int n, int i,
                                     int j) {
    if (j < 0 || j >= n) {
        return -1;
    }
    unsigned ki = morton_codes[i];
    unsigned kj = morton_codes[j];
    if (ki == kj) {
        // If Morton codes are equal, use index to break tie
        return 32 + clz(i ^ j);
    }
    return clz(ki ^ kj);
}

// Karras's algorithm: determine the range of keys covered by internal node i
// Returns (left, right) range and split position
__device__ void find_split(const unsigned *morton_codes, int n, int i,
                           int &left, int &right, int &split) {
    // Determine direction of the range
    int lcp_prev = longest_common_prefix(morton_codes, n, i, i - 1);
    int lcp_next = longest_common_prefix(morton_codes, n, i, i + 1);
    int d = (lcp_next > lcp_prev) ? 1 : -1;

    // Compute upper bound for the range length
    int lcp_min = longest_common_prefix(morton_codes, n, i, i - d);
    int l_max = 2;
    while (longest_common_prefix(morton_codes, n, i, i + l_max * d) > lcp_min) {
        l_max *= 2;
    }

    // Binary search to find the other end of the range
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (longest_common_prefix(morton_codes, n, i, i + (l + t) * d) >
            lcp_min) {
            l += t;
        }
    }
    int j = i + l * d;

    // Determine the range
    left = min(i, j);
    right = max(i, j);

    // Find split position using binary search
    int lcp_node = longest_common_prefix(morton_codes, n, left, right);
    int s = 0;
    int t = right - left;
    do {
        t = (t + 1) / 2;
        if (longest_common_prefix(morton_codes, n, left, left + s + t) >
            lcp_node) {
            s += t;
        }
    } while (t > 1);
    split = left + s;
}

// Initialize parent array with sentinel values
__global__ void init_parent_kernel(unsigned *parent, unsigned num_nodes) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        parent[i] = UINT_MAX; // Sentinel: no parent set yet
    }
}

// Build internal nodes in parallel using Karras's algorithm
// Leaves at 0..n-1, internal nodes at n..2n-2
__global__ void build_internal_nodes_kernel(const unsigned *morton_codes,
                                            const unsigned *sorted_indices,
                                            unsigned n, Vec2u *nodes) {

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    // First, set up leaf nodes
    if (i < n) {
        nodes[i] =
            Vec2u(sorted_indices[i] + 1, 0); // Leaf: primitive index + 1, 0
    }

    // Build internal nodes (n-1 internal nodes for n leaves)
    if (i < n - 1) {
        unsigned internal_idx = n + i; // Internal nodes start at index n

        int left, right, split;
        find_split(morton_codes, n, i, left, right, split);

        // Left child: if range [left, split] is single element, it's a leaf;
        // else internal
        unsigned left_child = (left == split) ? left : (n + split);
        // Right child: if range [split+1, right] is single element, it's a
        // leaf; else internal
        unsigned right_child = (split + 1 == right) ? right : (n + split + 1);

        nodes[internal_idx] = Vec2u(left_child + 1, right_child + 1);
    }
}

// Set parent pointers based on node children
__global__ void set_parent_kernel(const Vec2u *nodes, unsigned *parent,
                                  unsigned n, unsigned num_nodes) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        Vec2u node = nodes[i];
        // Internal nodes have node[1] != 0
        if (node[1] != 0) {
            unsigned left_child = node[0] - 1;
            unsigned right_child = node[1] - 1;
            parent[left_child] = i;
            parent[right_child] = i;
        }
    }
}

// Find root (node with parent == UINT_MAX) and set its parent to itself
__global__ void find_and_set_root_kernel(unsigned *parent, unsigned *root_idx,
                                         unsigned num_internal, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    // Check internal nodes only (indices n to n + num_internal - 1)
    if (i < num_internal) {
        unsigned node_idx = n + i;
        if (parent[node_idx] == UINT_MAX) {
            *root_idx = node_idx;
            parent[node_idx] = node_idx; // Root's parent is itself
        }
    }
}

// Swap node contents between two positions
__global__ void swap_nodes_kernel(Vec2u *nodes, unsigned idx1, unsigned idx2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Vec2u temp = nodes[idx1];
        nodes[idx1] = nodes[idx2];
        nodes[idx2] = temp;
    }
}

// Update all child references after a swap
__global__ void update_child_refs_kernel(Vec2u *nodes, unsigned num_nodes,
                                         unsigned old_idx, unsigned new_idx) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) {
        return;
    }

    Vec2u node = nodes[i];
    bool modified = false;

    // Check if left child references swapped nodes
    if (node[0] != 0) { // Has left child
        unsigned left = node[0] - 1;
        if (left == old_idx) {
            node[0] = new_idx + 1;
            modified = true;
        } else if (left == new_idx) {
            node[0] = old_idx + 1;
            modified = true;
        }
    }

    // Check if right child references swapped nodes
    if (node[1] != 0) { // Has right child (internal node)
        unsigned right = node[1] - 1;
        if (right == old_idx) {
            node[1] = new_idx + 1;
            modified = true;
        } else if (right == new_idx) {
            node[1] = old_idx + 1;
            modified = true;
        }
    }

    if (modified) {
        nodes[i] = node;
    }
}

// Update level data indices after a swap
__global__ void update_level_indices_kernel(unsigned *level_data, unsigned nnz,
                                            unsigned old_idx,
                                            unsigned new_idx) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }

    unsigned idx = level_data[i];
    if (idx == old_idx) {
        level_data[i] = new_idx;
    } else if (idx == new_idx) {
        level_data[i] = old_idx;
    }
}

// Compute depth of each node by walking up to root
__global__ void compute_depths_kernel(unsigned num_nodes, unsigned root_idx,
                                      const unsigned *parent, unsigned *depth) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) {
        return;
    }

    unsigned d = 0;
    unsigned current = i;
    while (current != root_idx) {
        current = parent[current];
        d++;
    }
    depth[i] = d;
}

// Find maximum depth
__global__ void find_max_depth_kernel(const unsigned *depth, unsigned n,
                                      unsigned *max_depth) {
    __shared__ unsigned shared_max[256];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;

    shared_max[tid] = (i < n) ? depth[i] : 0;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_depth, shared_max[0]);
    }
}

// Count nodes at each depth level
__global__ void count_levels_kernel(const unsigned *depth, unsigned n,
                                    unsigned *level_counts) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&level_counts[depth[i]], 1);
    }
}

// Scatter nodes to their level positions
__global__ void scatter_to_levels_kernel(const unsigned *depth, unsigned n,
                                         const unsigned *level_offsets,
                                         unsigned *level_positions,
                                         unsigned *level_data) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned d = depth[i];
        unsigned pos = atomicAdd(&level_positions[d], 1);
        level_data[level_offsets[d] + pos] = i;
    }
}

// Build tree and gather levels entirely on GPU
void build_tree_and_levels(unsigned n, Vec<Vec2u> &nodes,
                           VecVec<unsigned> &level,
                           const Vec<unsigned> &sorted_indices,
                           const Vec<unsigned> &morton_codes) {
    if (n == 0) {
        return;
    }

    unsigned num_nodes = n == 1 ? 1 : 2 * n - 1;

    // Verify pre-allocated storage is sufficient
    assert(num_nodes <= nodes.allocated &&
           "BVH nodes exceed pre-allocated size");
    assert(num_nodes <= storage::parent.allocated &&
           "Tree storage exceeds pre-allocated size");

    // Use pre-allocated storage - just update size
    nodes.size = num_nodes;

    const unsigned BLOCK = 256;
    unsigned grid_leaves = (n + BLOCK - 1) / BLOCK;
    unsigned grid_nodes = (num_nodes + BLOCK - 1) / BLOCK;

    // Handle single element case
    if (n == 1) {
        // Just set up the single leaf node
        build_internal_nodes_kernel<<<1, 1>>>(
            morton_codes.data, sorted_indices.data, n, nodes.data);
        CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

        // Single level with single node - use pre-allocated storage
        unsigned h_offsets[2] = {0, 1};
        unsigned h_data[1] = {0};
        CUDA_HANDLE_ERROR(cudaMemcpy(level.offset, h_offsets,
                                     2 * sizeof(unsigned),
                                     cudaMemcpyHostToDevice));
        CUDA_HANDLE_ERROR(cudaMemcpy(level.data, h_data, 1 * sizeof(unsigned),
                                     cudaMemcpyHostToDevice));
        level.size = 1;
        level.nnz = 1;
        return;
    }

    // Initialize parent array with sentinel values
    init_parent_kernel<<<grid_nodes, BLOCK>>>(storage::parent.data, num_nodes);

    // Build tree structure in parallel (nodes only, not parents)
    build_internal_nodes_kernel<<<grid_leaves, BLOCK>>>(
        morton_codes.data, sorted_indices.data, n, nodes.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Set parent pointers based on node children
    set_parent_kernel<<<grid_nodes, BLOCK>>>(nodes.data, storage::parent.data,
                                             n, num_nodes);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Find root (the internal node with no parent) and set its parent to itself
    // Use first element of max_depth_storage to store root index temporarily
    unsigned num_internal = n - 1;
    unsigned grid_internal = (num_internal + BLOCK - 1) / BLOCK;
    find_and_set_root_kernel<<<grid_internal, BLOCK>>>(
        storage::parent.data, storage::max_depth_storage.data, num_internal, n);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy root index to host
    unsigned root_idx;
    CUDA_HANDLE_ERROR(cudaMemcpy(&root_idx, storage::max_depth_storage.data,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));

    // Compute depth of each node (using original tree structure before swap)
    compute_depths_kernel<<<grid_nodes, BLOCK>>>(
        num_nodes, root_idx, storage::parent.data, storage::depth.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Find maximum depth - reuse max_depth_storage
    CUDA_HANDLE_ERROR(
        cudaMemset(storage::max_depth_storage.data, 0, sizeof(unsigned)));
    find_max_depth_kernel<<<grid_nodes, BLOCK>>>(
        storage::depth.data, num_nodes, storage::max_depth_storage.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
    unsigned max_depth_val;
    CUDA_HANDLE_ERROR(cudaMemcpy(&max_depth_val,
                                 storage::max_depth_storage.data,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));

    unsigned num_levels = max_depth_val + 1;

    // Verify level count doesn't exceed pre-allocated storage
    // MAX_LEVELS=64 supports balanced trees with up to 2^64 nodes
    // Degenerate trees are extremely rare with Morton-code-based construction
    assert(num_levels <= storage::MAX_LEVELS &&
           "Tree depth exceeds MAX_LEVELS - degenerate tree detected");

    // Count nodes at each level
    CUDA_HANDLE_ERROR(cudaMemset(storage::level_counts.data, 0,
                                 num_levels * sizeof(unsigned)));
    count_levels_kernel<<<grid_nodes, BLOCK>>>(storage::depth.data, num_nodes,
                                               storage::level_counts.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Compute level offsets (exclusive scan)
    std::vector<unsigned> h_counts(num_levels);
    CUDA_HANDLE_ERROR(cudaMemcpy(h_counts.data(), storage::level_counts.data,
                                 num_levels * sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    std::vector<unsigned> h_offsets(num_levels + 1);
    h_offsets[0] = 0;
    for (unsigned i = 0; i < num_levels; ++i) {
        h_offsets[i + 1] = h_offsets[i] + h_counts[i];
    }
    CUDA_HANDLE_ERROR(cudaMemcpy(storage::level_offsets.data, h_offsets.data(),
                                 (num_levels + 1) * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));

    // Scatter nodes to level positions - uses pre-allocated level storage
    CUDA_HANDLE_ERROR(cudaMemset(storage::level_positions.data, 0,
                                 num_levels * sizeof(unsigned)));
    scatter_to_levels_kernel<<<grid_nodes, BLOCK>>>(
        storage::depth.data, num_nodes, storage::level_offsets.data,
        storage::level_positions.data, level.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy offsets to level structure
    CUDA_HANDLE_ERROR(cudaMemcpy(level.offset, h_offsets.data(),
                                 (num_levels + 1) * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
    level.size = num_levels;
    level.nnz = num_nodes;

    // Now swap root to the expected position (num_nodes - 1) for query
    // compatibility This must happen AFTER level computation to preserve tree
    // structure for depth calculation
    unsigned target_idx = num_nodes - 1;
    if (root_idx != target_idx) {
        // First swap the node contents
        swap_nodes_kernel<<<1, 1>>>(nodes.data, root_idx, target_idx);
        CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

        // Then update all child references
        update_child_refs_kernel<<<grid_nodes, BLOCK>>>(nodes.data, num_nodes,
                                                        root_idx, target_idx);
        CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

        // Also update level data indices to reflect the swap
        unsigned grid_nnz = (level.nnz + BLOCK - 1) / BLOCK;
        update_level_indices_kernel<<<grid_nnz, BLOCK>>>(level.data, level.nnz,
                                                         root_idx, target_idx);
        CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

//==============================================================================
// AABB computation (level-by-level, no atomics)
//==============================================================================

// Compute leaf AABBs for faces
__global__ void
compute_leaf_aabbs_face_kernel(const Vec3f *x0, const Vec3f *x1,
                               float extrapolate, const Vec3u *face, unsigned n,
                               AABB *aabb, const FaceProp *prop,
                               const FaceParam *params, const Vec2u *nodes) {

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned leaf_idx = i;
    unsigned prim_idx = nodes[leaf_idx][0] - 1;

    Vec3u f = face[prim_idx];

    Vec3f y00 = x0[f[0]], y01 = x0[f[1]], y02 = x0[f[2]];
    Vec3f y10 = x1[f[0]], y11 = x1[f[1]], y12 = x1[f[2]];

    Vec3f z10 = float(extrapolate) * (y10 - y00) + y00;
    Vec3f z11 = float(extrapolate) * (y11 - y01) + y01;
    Vec3f z12 = float(extrapolate) * (y12 - y02) + y02;

    const FaceParam &fparam = params[prop[prim_idx].param_index];
    float margin = 0.5f * fparam.ghat + fparam.offset;

    AABB box;
    for (int d = 0; d < 3; ++d) {
        float minv0 = fminf(float(y00[d]), fminf(float(y01[d]), float(y02[d])));
        float maxv0 = fmaxf(float(y00[d]), fmaxf(float(y01[d]), float(y02[d])));
        float minv1 = fminf(float(z10[d]), fminf(float(z11[d]), float(z12[d])));
        float maxv1 = fmaxf(float(z10[d]), fmaxf(float(z11[d]), float(z12[d])));
        box.min[d] = float(fminf(minv0, minv1) - margin);
        box.max[d] = float(fmaxf(maxv0, maxv1) + margin);
    }
    aabb[leaf_idx] = box;
}

// Compute leaf AABBs for edges
__global__ void
compute_leaf_aabbs_edge_kernel(const Vec3f *x0, const Vec3f *x1,
                               float extrapolate, const Vec2u *edge, unsigned n,
                               AABB *aabb, const EdgeProp *prop,
                               const EdgeParam *params, const Vec2u *nodes) {

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned leaf_idx = i;
    unsigned prim_idx = nodes[leaf_idx][0] - 1;

    Vec2u e = edge[prim_idx];

    Vec3f y00 = x0[e[0]], y01 = x0[e[1]];
    Vec3f y10 = x1[e[0]], y11 = x1[e[1]];

    Vec3f z10 = float(extrapolate) * (y10 - y00) + y00;
    Vec3f z11 = float(extrapolate) * (y11 - y01) + y01;

    const EdgeParam &eparam = params[prop[prim_idx].param_index];
    float margin = 0.5f * eparam.ghat + eparam.offset;

    AABB box;
    for (int d = 0; d < 3; ++d) {
        float minv0 = fminf(float(y00[d]), float(y01[d]));
        float maxv0 = fmaxf(float(y00[d]), float(y01[d]));
        float minv1 = fminf(float(z10[d]), float(z11[d]));
        float maxv1 = fmaxf(float(z10[d]), float(z11[d]));
        box.min[d] = float(fminf(minv0, minv1) - margin);
        box.max[d] = float(fmaxf(maxv0, maxv1) + margin);
    }
    aabb[leaf_idx] = box;
}

// Compute leaf AABBs for vertices
__global__ void compute_leaf_aabbs_vertex_kernel(
    const Vec3f *x0, const Vec3f *x1, float extrapolate, unsigned n,
    AABB *aabb, const VertexProp *prop, const VertexParam *params,
    const Vec2u *nodes) {

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned leaf_idx = i;
    unsigned prim_idx = nodes[leaf_idx][0] - 1;

    Vec3f y0 = x0[prim_idx];
    Vec3f y1 = x1[prim_idx];
    Vec3f z1 = float(extrapolate) * (y1 - y0) + y0;

    const VertexParam &vparam = params[prop[prim_idx].param_index];
    float margin = 0.5f * vparam.ghat + vparam.offset;

    AABB box;
    for (int d = 0; d < 3; ++d) {
        float minv = fminf(float(y0[d]), float(z1[d]));
        float maxv = fmaxf(float(y0[d]), float(z1[d]));
        box.min[d] = float(minv - margin);
        box.max[d] = float(maxv + margin);
    }
    aabb[leaf_idx] = box;
}

// Merge children AABBs for internal nodes at a given level
__global__ void merge_aabbs_kernel(const unsigned *level_nodes,
                                   unsigned level_size, AABB *aabb,
                                   const Vec2u *nodes) {

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= level_size) {
        return;
    }

    unsigned node_idx = level_nodes[i];
    Vec2u node = nodes[node_idx];

    // Internal nodes have node[1] != 0
    if (node[1] == 0) {
        return;
    }

    unsigned left = node[0] - 1;
    unsigned right = node[1] - 1;

    AABB left_box = aabb[left];
    AABB right_box = aabb[right];

    AABB merged;
    for (int d = 0; d < 3; ++d) {
        merged.min[d] = left_box.min[d] < right_box.min[d] ? left_box.min[d]
                                                           : right_box.min[d];
        merged.max[d] = left_box.max[d] > right_box.max[d] ? left_box.max[d]
                                                           : right_box.max[d];
    }
    aabb[node_idx] = merged;
}

// Kernel to dispatch merge for a single level
__global__ void merge_level_kernel(const unsigned *level_data,
                                   const unsigned *level_offsets,
                                   unsigned level_idx, AABB *aabb,
                                   const Vec2u *nodes) {

    unsigned level_start = level_offsets[level_idx];
    unsigned level_size = level_offsets[level_idx + 1] - level_start;

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= level_size) {
        return;
    }

    unsigned node_idx = level_data[level_start + i];
    Vec2u node = nodes[node_idx];

    // Skip leaves (node[1] == 0)
    if (node[1] == 0) {
        return;
    }

    unsigned left = node[0] - 1;
    unsigned right = node[1] - 1;

    AABB left_box = aabb[left];
    AABB right_box = aabb[right];

    AABB merged;
    for (int d = 0; d < 3; ++d) {
        merged.min[d] = left_box.min[d] < right_box.min[d] ? left_box.min[d]
                                                           : right_box.min[d];
        merged.max[d] = left_box.max[d] > right_box.max[d] ? left_box.max[d]
                                                           : right_box.max[d];
    }
    aabb[node_idx] = merged;
}

// Propagate AABBs bottom-up using level information (all on GPU)
void propagate_aabbs(Vec<AABB> &aabb, const Vec<Vec2u> &nodes,
                     const VecVec<unsigned> &level) {
    if (level.size <= 1) {
        return; // Only leaves, nothing to propagate
    }

    const unsigned BLOCK = 256;

    // Copy offsets to host to determine grid sizes
    // (This is a small array, typically < 64 elements)
    std::vector<unsigned> h_offsets(level.size + 1);
    CUDA_HANDLE_ERROR(cudaMemcpy(h_offsets.data(), level.offset,
                                 (level.size + 1) * sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));

    // Level structure: level[0] = root (depth 0), level[max] = leaves (max
    // depth) For bottom-up propagation: process from leaves towards root Skip
    // level[level.size-1] (leaves - already computed by leaf kernel) Process
    // from level[level.size-2] down to level[0] (root)
    for (int l = (int)level.size - 2; l >= 0; --l) {
        unsigned level_size = h_offsets[l + 1] - h_offsets[l];
        if (level_size == 0) {
            continue;
        }
        unsigned grid = (level_size + BLOCK - 1) / BLOCK;
        merge_level_kernel<<<grid, BLOCK>>>(level.data, level.offset,
                                            (unsigned)l, aabb.data, nodes.data);
        // Synchronize after each level to ensure children are processed before
        // parents
        CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

//==============================================================================
// Main build functions
//==============================================================================

void build_face_bvh(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                    float extrapolate, const Vec<Vec3u> &face, BVH &bvh,
                    Vec<AABB> &aabb, const Vec<FaceProp> &prop,
                    const Vec<FaceParam> &params) {
    unsigned n = face.size;
    if (n == 0) {
        return;
    }
    assert(n <= storage::max_primitives &&
           "Face count exceeds pre-allocated LBVH storage");

    const unsigned BLOCK = 256;
    unsigned grid = (n + BLOCK - 1) / BLOCK;

    // 1. Compute centroids (from x1 for tree ordering)
    compute_face_centroids_kernel<<<grid, BLOCK>>>(
        x1.data, face.data, n, storage::centroids_x.data,
        storage::centroids_y.data, storage::centroids_z.data);

    // 2. Compute scene bounds
    init_bounds_kernel<<<1, 1>>>(storage::scene_bounds.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
    compute_scene_bounds_kernel<<<grid, BLOCK>>>(
        storage::centroids_x.data, storage::centroids_y.data,
        storage::centroids_z.data, n, storage::scene_bounds.data);

    // 3. Compute Morton codes
    compute_morton_codes_kernel<<<grid, BLOCK>>>(
        storage::centroids_x.data, storage::centroids_y.data,
        storage::centroids_z.data, n, storage::scene_bounds.data,
        storage::morton_codes.data, storage::sorted_indices.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 4. Sort by Morton code
    kernels::radix_sort_pairs(
        storage::morton_codes.data, storage::sorted_indices.data, n,
        storage::temp_morton.data, storage::temp_indices.data,
        storage::histogram_buffer.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 5. Build tree structure and gather levels entirely on GPU
    build_tree_and_levels(n, bvh.node, bvh.level, storage::sorted_indices,
                          storage::morton_codes);
    aabb.resize(bvh.node.size);

    // 6. Compute leaf AABBs
    compute_leaf_aabbs_face_kernel<<<grid, BLOCK>>>(
        x0.data, x1.data, extrapolate, face.data, n, aabb.data, prop.data,
        params.data, bvh.node.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 7. Propagate AABBs bottom-up level by level
    propagate_aabbs(aabb, bvh.node, bvh.level);
}

void build_edge_bvh(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                    float extrapolate, const Vec<Vec2u> &edge, BVH &bvh,
                    Vec<AABB> &aabb, const Vec<EdgeProp> &prop,
                    const Vec<EdgeParam> &params) {
    unsigned n = edge.size;
    if (n == 0) {
        return;
    }
    assert(n <= storage::max_primitives &&
           "Edge count exceeds pre-allocated LBVH storage");

    const unsigned BLOCK = 256;
    unsigned grid = (n + BLOCK - 1) / BLOCK;

    // 1. Compute centroids (from x1 for tree ordering)
    compute_edge_centroids_kernel<<<grid, BLOCK>>>(
        x1.data, edge.data, n, storage::centroids_x.data,
        storage::centroids_y.data, storage::centroids_z.data);

    // 2. Compute scene bounds
    init_bounds_kernel<<<1, 1>>>(storage::scene_bounds.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
    compute_scene_bounds_kernel<<<grid, BLOCK>>>(
        storage::centroids_x.data, storage::centroids_y.data,
        storage::centroids_z.data, n, storage::scene_bounds.data);

    // 3. Compute Morton codes
    compute_morton_codes_kernel<<<grid, BLOCK>>>(
        storage::centroids_x.data, storage::centroids_y.data,
        storage::centroids_z.data, n, storage::scene_bounds.data,
        storage::morton_codes.data, storage::sorted_indices.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 4. Sort by Morton code
    kernels::radix_sort_pairs(
        storage::morton_codes.data, storage::sorted_indices.data, n,
        storage::temp_morton.data, storage::temp_indices.data,
        storage::histogram_buffer.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 5. Build tree structure and gather levels entirely on GPU
    build_tree_and_levels(n, bvh.node, bvh.level, storage::sorted_indices,
                          storage::morton_codes);
    aabb.resize(bvh.node.size);

    // 6. Compute leaf AABBs
    compute_leaf_aabbs_edge_kernel<<<grid, BLOCK>>>(
        x0.data, x1.data, extrapolate, edge.data, n, aabb.data, prop.data,
        params.data, bvh.node.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 7. Propagate AABBs bottom-up level by level
    propagate_aabbs(aabb, bvh.node, bvh.level);
}

void build_vertex_bvh(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                      float extrapolate, BVH &bvh, Vec<AABB> &aabb,
                      unsigned surface_vert_count, const Vec<VertexProp> &prop,
                      const Vec<VertexParam> &params) {
    unsigned n = surface_vert_count;
    if (n == 0) {
        return;
    }
    assert(n <= storage::max_primitives &&
           "Vertex count exceeds pre-allocated LBVH storage");

    const unsigned BLOCK = 256;
    unsigned grid = (n + BLOCK - 1) / BLOCK;

    // 1. Compute centroids (from x1 for tree ordering)
    compute_vertex_centroids_kernel<<<grid, BLOCK>>>(
        x1.data, n, storage::centroids_x.data, storage::centroids_y.data,
        storage::centroids_z.data);

    // 2. Compute scene bounds
    init_bounds_kernel<<<1, 1>>>(storage::scene_bounds.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
    compute_scene_bounds_kernel<<<grid, BLOCK>>>(
        storage::centroids_x.data, storage::centroids_y.data,
        storage::centroids_z.data, n, storage::scene_bounds.data);

    // 3. Compute Morton codes
    compute_morton_codes_kernel<<<grid, BLOCK>>>(
        storage::centroids_x.data, storage::centroids_y.data,
        storage::centroids_z.data, n, storage::scene_bounds.data,
        storage::morton_codes.data, storage::sorted_indices.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 4. Sort by Morton code
    kernels::radix_sort_pairs(
        storage::morton_codes.data, storage::sorted_indices.data, n,
        storage::temp_morton.data, storage::temp_indices.data,
        storage::histogram_buffer.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 5. Build tree structure and gather levels entirely on GPU
    build_tree_and_levels(n, bvh.node, bvh.level, storage::sorted_indices,
                          storage::morton_codes);
    aabb.resize(bvh.node.size);

    // 6. Compute leaf AABBs
    compute_leaf_aabbs_vertex_kernel<<<grid, BLOCK>>>(
        x0.data, x1.data, extrapolate, n, aabb.data, prop.data, params.data,
        bvh.node.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // 7. Propagate AABBs bottom-up level by level
    propagate_aabbs(aabb, bvh.node, bvh.level);
}

//==============================================================================
// AABB-only update functions (reuse existing tree structure)
//==============================================================================

void update_face_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                      float extrapolate, const Vec<Vec3u> &face, const BVH &bvh,
                      Vec<AABB> &aabb, const Vec<FaceProp> &prop,
                      const Vec<FaceParam> &params) {
    unsigned n = face.size;
    if (n == 0) {
        return;
    }

    const unsigned BLOCK = 256;
    unsigned grid = (n + BLOCK - 1) / BLOCK;

    // Compute leaf AABBs
    compute_leaf_aabbs_face_kernel<<<grid, BLOCK>>>(
        x0.data, x1.data, extrapolate, face.data, n, aabb.data, prop.data,
        params.data, bvh.node.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Propagate AABBs bottom-up level by level
    propagate_aabbs(aabb, bvh.node, bvh.level);
}

void update_edge_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                      float extrapolate, const Vec<Vec2u> &edge, const BVH &bvh,
                      Vec<AABB> &aabb, const Vec<EdgeProp> &prop,
                      const Vec<EdgeParam> &params) {
    unsigned n = edge.size;
    if (n == 0) {
        return;
    }

    const unsigned BLOCK = 256;
    unsigned grid = (n + BLOCK - 1) / BLOCK;

    // Compute leaf AABBs
    compute_leaf_aabbs_edge_kernel<<<grid, BLOCK>>>(
        x0.data, x1.data, extrapolate, edge.data, n, aabb.data, prop.data,
        params.data, bvh.node.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Propagate AABBs bottom-up level by level
    propagate_aabbs(aabb, bvh.node, bvh.level);
}

void update_vertex_aabb(const Vec<Vec3f> &x0, const Vec<Vec3f> &x1,
                        float extrapolate, const BVH &bvh, Vec<AABB> &aabb,
                        unsigned surface_vert_count,
                        const Vec<VertexProp> &prop,
                        const Vec<VertexParam> &params) {
    unsigned n = surface_vert_count;
    if (n == 0) {
        return;
    }

    const unsigned BLOCK = 256;
    unsigned grid = (n + BLOCK - 1) / BLOCK;

    // Compute leaf AABBs
    compute_leaf_aabbs_vertex_kernel<<<grid, BLOCK>>>(
        x0.data, x1.data, extrapolate, n, aabb.data, prop.data, params.data,
        bvh.node.data);
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    // Propagate AABBs bottom-up level by level
    propagate_aabbs(aabb, bvh.node, bvh.level);
}

void build_collision_mesh_bvh(const DataSet &data, const ParamSet &param) {
    const CollisionMesh &mesh = data.constraint.mesh;
    BVHSet &bvh = storage::collision_mesh_bvh;

    // Build face BVH for collision mesh (static - no motion)
    if (mesh.face.size > 0) {
        build_face_bvh(mesh.vertex, mesh.vertex, 1.0f, mesh.face, bvh.face,
                       contact::get_collision_mesh_face_aabb(), mesh.prop.face,
                       mesh.param_arrays.face);
    }

    // Build edge BVH for collision mesh (static - no motion)
    if (mesh.edge.size > 0) {
        build_edge_bvh(mesh.vertex, mesh.vertex, 1.0f, mesh.edge, bvh.edge,
                       contact::get_collision_mesh_edge_aabb(), mesh.prop.edge,
                       mesh.param_arrays.edge);
    }
}

} // namespace lbvh

// BVH storage accessors
namespace bvh_storage {

BVHSet &get_bvh() { return lbvh::storage::main_bvh; }

BVHSet &get_collision_mesh_bvh() { return lbvh::storage::collision_mesh_bvh; }

} // namespace bvh_storage
