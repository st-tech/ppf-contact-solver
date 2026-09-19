// File: lbvh.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The two facts a backend cannot infer are written as
// C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::device]]` is the address space of a pointer parameter. MSL requires
// the second on every pointer and reference type; CUDA and the host have one
// address space and are handed the same declarations with it removed.
//
// Every pointer below reads GPU global memory, the sorted Morton codes and the
// parent links of the tree under construction, so every one of them is
// `[[seam::device]]`. Nothing here is passed by reference and nothing reads
// threadgroup memory.
//
// The arithmetic goes through the `ppf` namespace, which each backend prologue
// defines with that backend's spelling: `fmath::min` and `fmath::max` are the
// float forms rather than the promoting overload set, and `bits::clz` is the
// count of leading zeros the radix split is built on.

[[seam::device_fn]] inline unsigned lbvh_expand_bits(unsigned value) {
    value = (value | (value << 16)) & 0x030000FFu;
    value = (value | (value << 8)) & 0x0300F00Fu;
    value = (value | (value << 4)) & 0x030C30C3u;
    value = (value | (value << 2)) & 0x09249249u;
    return value;
}

[[seam::device_fn]] inline unsigned
lbvh_morton_code_3d(unsigned x, unsigned y, unsigned z) {
    return lbvh_expand_bits(x) | (lbvh_expand_bits(y) << 1) |
           (lbvh_expand_bits(z) << 2);
}

[[seam::device_fn]] inline unsigned
lbvh_quantize(float value, float minimum, float maximum) {
    const float extent = maximum - minimum;
    const float scale = extent > 1.0e-10f ? 1023.0f / extent : 0.0f;
    return static_cast<unsigned>(
        fmath::min(1023.0f, fmath::max(0.0f, (value - minimum) * scale)));
}

// THE BOUNDS ARRIVE AS AN ARRAY, six floats in the order the reduction writes
// them: three minima then three maxima. `compute_morton_codes_kernel` reads
// `bounds[0..5]` the same way, and for the same reason: the pass that produces
// them runs on the device, so a caller that had to name them as scalars would
// owe a download between the two dispatches.
[[seam::device_fn]] inline unsigned lbvh_morton_from_bounds(
    float x, float y, float z, const float *bounds) {
    return lbvh_morton_code_3d(
        lbvh_quantize(x, bounds[0], bounds[3]),
        lbvh_quantize(y, bounds[1], bounds[4]),
        lbvh_quantize(z, bounds[2], bounds[5]));
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `lbvh_morton_from_bounds_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// NO COMPOSITION SITS BETWEEN THIS AND THE BODY, which is what makes it the
// cheapest shape a conversion can take: three element gathers, six scene-wide
// scalars and one element scatter are exactly the arguments the body already
// takes, in the order it takes them, so the generated statement is the range
// shim's `codes[i] = lbvh_morton_from_bounds(cx[i], cy[i], cz[i], ...)`
// character for character.
//
// THE SIX BOUNDS ARE SCALARS RATHER THAN A BUFFER because they are the SCENE
// bounds, one box the caller reduced before the dispatch and every thread reads
// identically. They quantize the code, so a thread reading a different box would
// place its primitive in a different cell.
//
// AND SIX SCALARS RATHER THAN TWO TRIPLES, which is a record-layout rule and not
// a matter of taste: a three-float member is a `float3` on Metal, 16 bytes
// against 12, and a record's layout is the one thing every backend must agree
// on. Never put a three-float vector in a record that crosses this seam.
//
// A MORTON CODE DECIDES TREE TOPOLOGY, so this pass is not free to be
// reassociated: `sort.rs` resolves a duplicate-code run by primitive index
// because a permuted run yields a different tree, and that contract starts with
// the code this body returns.
[[seam::entry(count)]] void lbvh_morton_from_bounds(
    const float *cx,
    const float *cy,
    const float *cz,
    const float *bounds,
    unsigned *codes,
    unsigned count);

struct LbvhSplit {
    int left;
    int right;
    int split;
};

struct LbvhNode {
    unsigned first;
    unsigned second;
};

[[seam::device_fn]] inline int lbvh_longest_common_prefix(
    const unsigned *morton_codes, int count, int i, int j) {
    if (j < 0 || j >= count) {
        return -1;
    }
    const unsigned a = morton_codes[i];
    const unsigned b = morton_codes[j];
    if (a == b) {
        return 32 + (i == j ? 32 : bits::clz(static_cast<unsigned>(i ^ j)));
    }
    return bits::clz(a ^ b);
}

[[seam::device_fn]] inline LbvhSplit lbvh_find_split(
    const unsigned *morton_codes, int count, int i) {
    const int prefix_previous =
        lbvh_longest_common_prefix(morton_codes, count, i, i - 1);
    const int prefix_next =
        lbvh_longest_common_prefix(morton_codes, count, i, i + 1);
    const int direction = prefix_next > prefix_previous ? 1 : -1;
    const int prefix_minimum = lbvh_longest_common_prefix(
        morton_codes, count, i, i - direction);
    int maximum_length = 2;
    while (lbvh_longest_common_prefix(
               morton_codes, count, i,
               i + maximum_length * direction) > prefix_minimum) {
        maximum_length *= 2;
    }

    int length = 0;
    for (int step = maximum_length / 2; step >= 1; step /= 2) {
        if (lbvh_longest_common_prefix(
                morton_codes, count, i,
                i + (length + step) * direction) > prefix_minimum) {
            length += step;
        }
    }
    const int j = i + length * direction;
    LbvhSplit result;
    result.left = i < j ? i : j;
    result.right = i > j ? i : j;
    const int node_prefix = lbvh_longest_common_prefix(
        morton_codes, count, result.left, result.right);
    int split_offset = 0;
    int step = result.right - result.left;
    do {
        step = (step + 1) / 2;
        if (lbvh_longest_common_prefix(
                morton_codes, count, result.left,
                result.left + split_offset + step) > node_prefix) {
            split_offset += step;
        }
    } while (step > 1);
    result.split = result.left + split_offset;
    return result;
}

[[seam::device_fn]] inline LbvhNode
lbvh_leaf_node(unsigned primitive_index) {
    return LbvhNode{primitive_index + 1u, 0u};
}

// The inverse of lbvh_leaf_node's encoding: the primitive a leaf stands
// for, read back out of the node's first child slot.
//
// A node stores its children BIASED BY ONE so that zero can mean "no child",
// which is what makes a leaf recognizable at all (its second slot is 0). The
// bias is therefore part of the node format rather than an implementation
// detail, and reading it back by hand at a call site is a second statement of
// the format that can drift from this one.
[[seam::device_fn]] inline unsigned
lbvh_leaf_primitive(unsigned first_child) {
    return first_child - 1u;
}

[[seam::device_fn]] inline LbvhNode lbvh_internal_node(
    const unsigned *morton_codes, unsigned count, unsigned i) {
    const LbvhSplit range = lbvh_find_split(
        morton_codes, static_cast<int>(count), static_cast<int>(i));
    const unsigned left =
        range.left == range.split ? static_cast<unsigned>(range.left)
                                  : count + static_cast<unsigned>(range.split);
    const unsigned right =
        range.split + 1 == range.right
            ? static_cast<unsigned>(range.right)
            : count + static_cast<unsigned>(range.split + 1);
    return LbvhNode{left + 1u, right + 1u};
}

// ONE THREAD'S SHARE OF THE TREE: the leaf at `i` and the internal node above
// it, which is why one dispatch over `[0, n)` writes both classes.
//
// TWO CONDITIONAL WRITES AT DIFFERENT INDEX ARITHMETIC, and that is what makes
// `nodes` a BASE pointer rather than a scatter. A scatter carries the body's one
// return value to the thread's own slot; this writes four slots at two
// unrelated offsets, `2 * i` for the leaf and `2 * (n + i)` for the internal
// node, and writes neither when its guard fails. The guards are not the
// dispatch's: the extent is `n` while the leaf class covers `[0, n)` and the
// internal class `[0, n - 1)`, so the last thread writes a leaf and no internal
// node.
// `primitive_count` is the tree's own `n`, which the body needs to place an
// internal node at `n + i` and to bound the split search. The dispatch extent is
// also `n`, and the two are separate because the body reads `n` as the tree's
// SIZE rather than as its own bound.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lbvh_nodes(
    const unsigned *morton,
    const unsigned *sorted, unsigned primitive_count,
    unsigned *nodes, unsigned i) {
    if (i < primitive_count) {
        const LbvhNode leaf = lbvh_leaf_node(sorted[i]);
        nodes[2 * i] = leaf.first;
        nodes[2 * i + 1] = leaf.second;
    }
    if (i + 1 < primitive_count) {
        const LbvhNode internal =
            lbvh_internal_node(morton, primitive_count, i);
        nodes[2 * (primitive_count + i)] = internal.first;
        nodes[2 * (primitive_count + i) + 1] = internal.second;
    }
}

// THE DEPTH PASS'S ENTRY POINT, and the shape it takes is the one a walk needs.
// `parent` stays a BASE pointer because the body climbs it from an arbitrary
// node rather than reading a fixed run: a gather would hand the body one link
// and the walk needs the whole array, so the addressing is the body's business
// and the thread index is forwarded instead of spent on it.
//
// THE ROOT IS A SCALAR because it is the tree's, one value every thread
// compares against, and the depth it yields is one word per node, which is
// exactly `[[seam::scatter]]` over a value-returning body.
//
// NO BOUND IS DECLARABLE HERE and that is a property of the walk rather than an
// omission: `[[seam::bound]]` checks the slots of an index LIST, and this body
// holds no list. It reads `parent[node]` for a node it computed itself, so the
// index space it walks is the one the caller allocated `parent` over. A tree
// whose parent links do not reach the root does not fail a bound, it fails to
// terminate, which is the invariant `lbvh.rs` establishes when it builds them.
[[seam::entry(node, depth)]]
[[seam::device_fn]] inline unsigned lbvh_node_depth(
    const unsigned *parent,
    const unsigned *root, unsigned node) {
    const unsigned target = root[0];
    unsigned depth = 0;
    while (node != target) {
        node = parent[node];
        ++depth;
    }
    return depth;
}

// ---------------------------------------------------------------------------
// THE MORTON SORT KEY OF EACH PRIMITIVE: the centroid of its vertices, in
// ABSOLUTE world coordinates.
//
// ABSOLUTE BY CONTRACT, and this is one of the few places that is right. A
// Morton code is a QUANTIZATION OF WORLD SPACE, so the absolute magnitude IS
// the signal; a translation-invariant difference would carry none.
//
// THE THREE COMPONENTS GO TO THREE SEPARATE ARRAYS rather than one interleaved
// triple, because that is the layout the Morton pass reads and because a
// three-float member renders as `float3` on Metal, 16 bytes against 12. They
// are `[[seam::stride(1)]]` pointers to this element's own slot for the reason
// `main/velocity.kernel.cpp` gives: one call returns one value, and a body with
// three outputs has none of them as the return a `[[seam::scatter]]` carries.
//
// THE POSITION BUFFER IS SPELLED `vert`, NOT `vertex`, because `vertex` is a
// shader-stage qualifier in MSL. The generator refuses it by name. A
// hand-written shim never meets that rule, so the spelling differs between the
// two forms for a reason that is not stylistic.
//
// `vert` is `[[seam::through]]` the `[[seam::indices(N)]]` list, so the entry
// reads the element's N slots, checks each against `vertex_count` and hands the
// body the N positions they name. THE BOUND IS WHAT THAT ADDS: a slot is data
// rather than the thread index, so `[[seam::count]]` says nothing about it, and
// Metal returns 0.0 for an out-of-bounds read rather than faulting, which would
// turn a corrupt index list into a plausible centroid and so a plausible tree.

[[seam::device_fn]] inline void face_centroid(
    const Vec3f &v0, const Vec3f &v1,
    const Vec3f &v2, float *cx,
    float *cy, float *cz) {
    *cx = (v0[0] + v1[0] +
           v2[0]) / 3.0f;
    *cy = (v0[1] + v1[1] +
           v2[1]) / 3.0f;
    *cz = (v0[2] + v1[2] +
           v2[2]) / 3.0f;
}

[[seam::entry(count)]] void face_centroid(
    [[seam::through]] const Vec3f *vert,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    [[seam::stride(1)]] float *cx,
    [[seam::stride(1)]] float *cy,
    [[seam::stride(1)]] float *cz,
    unsigned count);

[[seam::device_fn]] inline void edge_centroid(
    const Vec3f &v0, const Vec3f &v1,
    float *cx, float *cy,
    float *cz) {
    *cx = (v0[0] + v1[0]) / 2.0f;
    *cy = (v0[1] + v1[1]) / 2.0f;
    *cz = (v0[2] + v1[2]) / 2.0f;
}

[[seam::entry(count)]] void edge_centroid(
    [[seam::through]] const Vec3f *vert,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    [[seam::stride(1)]] float *cx,
    [[seam::stride(1)]] float *cy,
    [[seam::stride(1)]] float *cz,
    unsigned count);

// A vertex is its own centroid, so this one gathers nothing and needs no bound:
// the element index IS the slot, which `[[seam::count]]` already guards.
[[seam::device_fn]] inline void vertex_centroid(
    const Vec3f &v, float *cx,
    float *cy, float *cz) {
    *cx = v[0];
    *cy = v[1];
    *cz = v[2];
}

[[seam::entry(count)]] void vertex_centroid(
    const Vec3f *vert,
    [[seam::stride(1)]] float *cx,
    [[seam::stride(1)]] float *cy,
    [[seam::stride(1)]] float *cz,
    unsigned count);

// ---------------------------------------------------------------------------
// THE TREE'S PARENT LINKS, ROOT AND LEVELS, all four on the device.
//
// Each of the four is an entry point below: `lbvh_set_parent`,
// `lbvh_find_root`, `lbvh_count_levels` and `lbvh_scatter_levels`. A host
// spelling would download the whole node array and fold it there, and calling
// such a fold "index bookkeeping rather than a kernel" is not a reason to move
// it: a pass whose work is proportional to the node count stays on the device,
// whatever the clock says.
//
// WHAT CROSSES TO THE HOST IS BOUNDED BY THE LEVEL COUNT, NOT BY THE NODES: the
// root and its count, and the per-level counts, which the host scans over
// `num_levels` entries.

// Each internal node claims its two children.
//
// DISJOINT WITHOUT AN ATOMIC, because a child has exactly one parent: the two
// writes below are the only writes to those two slots in the whole pass. A leaf
// is the second slot being zero, which is how `lbvh_nodes` encodes it.
[[seam::entry(node)]]
[[seam::device_fn]] inline void lbvh_set_parent(
    const unsigned *nodes,
    unsigned *parent, unsigned node) {
    const unsigned second = nodes[2u * node + 1u];
    if (second == 0u) {
        return;
    }
    parent[nodes[2u * node] - 1u] = node;
    parent[second - 1u] = node;
}

// The root is the internal node no other node points at.
//
// FOUND RATHER THAN ASSUMED, because Karras's construction does not place it at
// a fixed index. `found` is a COUNTER beside the index and not decoration: the
// host asserted "the tree has two roots" over its serial scan, and that
// guarantee has to survive the move to a kernel where no thread can see
// another's verdict. The host reads the counter and fails if it is not one.
[[seam::entry(internal)]]
[[seam::device_fn]] inline void lbvh_find_root(
    unsigned *parent, unsigned primitive_count,
    unsigned *root,
    compute::atomic_uint_t *found, unsigned internal) {
    const unsigned node = primitive_count + internal;
    if (parent[node] != 0xffffffffu) {
        return;
    }
    // The root's parent is ITSELF, which is what makes `lbvh_node_depth`'s walk
    // terminate rather than reading the sentinel as a node index.
    parent[node] = node;
    root[0] = node;
    compute::atomic_add(found, 1u);
}

// One node's depth, counted into its level's bin.
//
// `capacity` IS CHECKED rather than assumed. A degenerate tree deeper than the
// bins allocated for it would otherwise write past them, which Metal drops
// silently and reports as success, and the level pass below would then place
// nodes at offsets nothing wrote.
template <typename D>
[[seam::device_fn]] inline void lbvh_count_levels(
    const unsigned *depth,
    compute::atomic_uint_t *counts, unsigned capacity,
    unsigned node, D diag) {
    const unsigned d = depth[node];
    DIAG_ASSERT4(diag, d < capacity, static_cast<float>(d),
                static_cast<float>(capacity), static_cast<float>(node), 0.0f);
    if (d >= capacity) {
        return;
    }
    compute::atomic_add(counts + d, 1u);
}

// One node into its level's run.
//
// THE ORDER WITHIN A LEVEL IS NOT FIXED, and it does not need to be: a level's
// nodes are independent, each box depending only on children a level deeper, so
// the propagation reads the same values whatever order the run holds them in.
// Each node claims its slot with one atomic increment of its level's cursor.
[[seam::device_fn]] inline void lbvh_scatter_levels(
    const unsigned *depth,
    const unsigned *level_offset,
    compute::atomic_uint_t *cursor,
    unsigned *level_data, unsigned node) {
    const unsigned d = depth[node];
    const unsigned slot = compute::atomic_add(cursor + d, 1u);
    level_data[level_offset[d] + slot] = node;
}

[[seam::entry(count, node)]] void lbvh_count_levels(
    const unsigned *depth,
    compute::atomic_uint_t *counts,
    unsigned capacity,
    unsigned node,
    DiagHandle diag,
    unsigned count);

[[seam::entry(count, node)]] void lbvh_scatter_levels(
    const unsigned *depth,
    const unsigned *level_offset,
    compute::atomic_uint_t *cursor,
    unsigned *level_data,
    unsigned node,
    unsigned count);
