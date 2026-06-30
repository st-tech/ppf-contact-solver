// File: main.cpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// libsimbackend_cpu: the CUDA-free emulator. Implements the same
// extern "C" FFI as cpp/main/main.cu, but every kernel is replaced
// with a no-op and every cuda* call is replaced with the std:: CPU
// equivalent (see mem.hpp).
//
// Behavior contract:
//   * advance() does no physics work. It bumps param->time, copies
//     it to result.time, sleeps PPF_EMULATED_STEP_MS so the test rig
//     can observe BUSY/RUNNING transitions, and runs the optional
//     PPF_EMULATED_FAIL_AT_FRAME fault-injection branch.
//   * update_constraint() takes over the role the kernel plays in
//     production: kinematic FixPair positions are written directly
//     into the device-side vertex.curr buffer so the next fetch()
//     round-trip yields a curr_vertex with pinned vertices at their
//     kinematic targets.
//   * fetch() / fetch_inv_rest() / fetch_rest_angles() are real
//     memcpy round-trips, identical-shape to the CUDA versions.
//   * fetch_dyn_counts/fetch_dyn/update_dyn/init_collision_windows/
//     refresh_collision_active are no-ops modulo state mirroring (no
//     contact assembly happens).
//   * override_velocity/override_angular_velocity/gather_current_positions
//     are real: they read/bias the vertex.curr/prev buffers so the optional
//     PPF_EMULATED_ELASTIC step (pd_arap) picks up the injected velocity.
//     With the elastic solver off they have no observable effect (advance()
//     never integrates prev).
//
// Stripped Rust-side stubs (cuda_stubs, apply_kinematic_constraint,
// emulated_intersection, time-bump+sleep) live here now.

#include "../cpp/data.hpp"
#include "mem.hpp"
#include "pd_arap.hpp"
#include "sand.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// The C++ TU is now linked statically into the Rust binary on every
// platform (see build.rs). The DLL_EXPORT macro stays so the symbols
// are visible to the linker even if the static archive layout
// changes.
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

namespace {

DataSet g_host_dataset{};
DataSet g_dev_dataset{};
ParamSet *g_param = nullptr;
std::atomic<unsigned long long> g_advance_count{0};

// PPF_EMULATED_FAIL_AT_FRAME parsing. Cached on first read so the env
// var only counts at start-of-run. Negative or unset means disabled.
int parse_fail_at_frame() {
    static int cached = -2;
    if (cached != -2) {
        return cached;
    }
    const char *raw = std::getenv("PPF_EMULATED_FAIL_AT_FRAME");
    if (!raw || !*raw) {
        cached = -1;
        return cached;
    }
    char *end = nullptr;
    long n = std::strtol(raw, &end, 10);
    if (end == raw || n < 0) {
        cached = -1;
    } else {
        cached = static_cast<int>(n);
    }
    return cached;
}

unsigned long long parse_step_ms() {
    static unsigned long long cached = ~0ull;
    if (cached != ~0ull) {
        return cached;
    }
    const char *raw = std::getenv("PPF_EMULATED_STEP_MS");
    if (!raw || !*raw) {
        cached = 1000;
        return cached;
    }
    char *end = nullptr;
    long long ms = std::strtoll(raw, &end, 10);
    cached = (end == raw || ms < 0) ? 1000 : static_cast<unsigned long long>(ms);
    return cached;
}

// Synthetic intersection records emitted when PPF_EMULATED_FAIL_AT_FRAME
// fires. Mirrors the layout that write_intersection_records in
// backend.rs reads: itype, elem0, elem1, num_verts0, num_verts1, then
// 5 packed vec3 positions.
std::vector<IntersectionRecord> g_synthetic_records;

void seed_synthetic_records() {
    if (!g_synthetic_records.empty()) {
        return;
    }
    auto make_record = [](unsigned itype, unsigned elem0, unsigned elem1,
                          unsigned nv0, unsigned nv1,
                          const float (&pos)[15]) {
        IntersectionRecord r{};
        r.type = itype;
        r.elem0 = elem0;
        r.elem1 = elem1;
        r.num_verts0 = nv0;
        r.num_verts1 = nv1;
        std::memcpy(r.positions, pos, sizeof(pos));
        return r;
    };

    // itype 0: face-edge (3-vert face, 2-vert edge).
    float fe_pos[15] = {0.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.5f, 0.5f, -0.1f,
                        0.5f, 0.5f, 0.1f};
    g_synthetic_records.push_back(make_record(0, 11, 22, 3, 2, fe_pos));

    // itype 1: edge-edge (2-vert edge, 2-vert edge).
    float ee_pos[15] = {0.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f,
                        0.5f, -0.5f, 0.0f,
                        0.5f, 0.5f, 0.0f,
                        0.0f, 0.0f, 0.0f};
    g_synthetic_records.push_back(make_record(1, 33, 44, 2, 2, ee_pos));

    // itype 2: collision-mesh (3-vert face, 2-vert edge), same shape
    // as the production recorder (5 vec3s max).
    float cm_pos[15] = {0.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.5f, 0.5f, -0.2f,
                        0.5f, 0.5f, 0.2f};
    g_synthetic_records.push_back(make_record(2, 55, 66, 3, 2, cm_pos));
}

// CPU malloc-backed mirror of the host dataset. Mirrors only the
// fields the FFI surface touches (vertex.curr/prev, prop.vertex/edge/
// face/hinge/tet for fix-flag rebuild, inv_rest2x2/3x3, constraint).
// Other fields stay nullptr in dev_dataset.
DataSet build_dev_mirror(const DataSet &host) {
    DataSet dev = host;

    dev.vertex.prev = mem::malloc_device(host.vertex.prev);
    dev.vertex.curr = mem::malloc_device(host.vertex.curr);

    dev.prop.vertex = mem::malloc_device(host.prop.vertex);
    dev.prop.edge = mem::malloc_device(host.prop.edge);
    dev.prop.face = mem::malloc_device(host.prop.face);
    dev.prop.hinge = mem::malloc_device(host.prop.hinge);
    dev.prop.tet = mem::malloc_device(host.prop.tet);
    dev.prop.pdrd_body = mem::malloc_device(host.prop.pdrd_body);
    dev.pdrd_vert_list = mem::malloc_device(host.pdrd_vert_list);
    dev.pdrd_rest_centered = mem::malloc_device(host.pdrd_rest_centered);
    dev.grain_omega = mem::malloc_device(host.grain_omega);
    dev.grain_inv_inertia = mem::malloc_device(host.grain_inv_inertia);
    dev.grain_torque = mem::malloc_device(host.grain_torque);
    dev.grain_ang_stiff = mem::malloc_device(host.grain_ang_stiff);
    dev.grain_contact_normal = mem::malloc_device(host.grain_contact_normal);
    dev.grain_inv_inertia_center =
        mem::malloc_device(host.grain_inv_inertia_center);
    dev.grain_omega_prev = mem::malloc_device(host.grain_omega_prev);
    dev.grain_A = mem::malloc_device(host.grain_A);
    dev.grain_B = mem::malloc_device(host.grain_B);
    dev.grain_grot = mem::malloc_device(host.grain_grot);

    dev.inv_rest2x2 = mem::malloc_device(host.inv_rest2x2);
    dev.inv_rest3x3 = mem::malloc_device(host.inv_rest3x3);

    dev.constraint = host.constraint;
    dev.constraint.fix = mem::malloc_device(host.constraint.fix);
    dev.constraint.pull = mem::malloc_device(host.constraint.pull);
    dev.constraint.torque_groups =
        mem::malloc_device(host.constraint.torque_groups);
    dev.constraint.torque_vertices =
        mem::malloc_device(host.constraint.torque_vertices);
    dev.constraint.sphere = mem::malloc_device(host.constraint.sphere);
    dev.constraint.floor = mem::malloc_device(host.constraint.floor);
    dev.constraint.stitch = mem::malloc_device(host.constraint.stitch);

    return dev;
}

} // namespace

// =============================================================
// extern "C" FFI: identical signatures to cpp/main/main.cu
// =============================================================

extern "C" DLL_EXPORT void set_log_path(const char * /*data_dir*/) {
    // No-op in the emulator. SimpleLog isn't compiled in.
}

// The emulator has no CUDA error / exit(1) fatal paths, so it always
// reports "no fatal exit". Present so the host's FFI symbol resolves on
// the emulated link, matching cpp/main/main.cu.
extern "C" DLL_EXPORT unsigned char ppf_fatal_code() {
    return 0;
}

extern "C" DLL_EXPORT bool initialize(DataSet *dataset, ParamSet *param) {
    g_host_dataset = *dataset;
    g_dev_dataset = build_dev_mirror(*dataset);
    g_param = param;
    g_advance_count.store(0);
    return true;
}

extern "C" DLL_EXPORT void advance(StepResult *result) {
    result->ccd_success = true;
    result->pcg_success = true;
    result->intersection_free = true;

    unsigned long long n_calls = g_advance_count.fetch_add(1) + 1;

    int fail_at = parse_fail_at_frame();
    if (fail_at >= 0 &&
        n_calls > static_cast<unsigned long long>(fail_at) + 1) {
        seed_synthetic_records();
        result->intersection_free = false;
    }

    // Optional implicit ARAP elastic step (opt-in via PPF_EMULATED_ELASTIC).
    // Default-off preserves the historical kinematic-only emulator: free
    // vertices move only when this solver runs; pins are already written
    // into vertex.curr by update_constraint() and act as Dirichlet targets.
    if (g_param && pd_arap::enabled()) {
        pd_arap::step(g_dev_dataset, *g_param);
    }

    // Phony granular mover. Auto-runs for a faceless/edgeless point cloud
    // (sand::step self-gates via sand::is_point_cloud) and is a no-op for any
    // scene with elements, so no opt-in flag is needed.
    if (g_param) {
        sand::step(g_dev_dataset, *g_param);
    }

    if (g_param) {
        float playback = g_param->playback != 0.0f ? g_param->playback : 1.0f;
        g_param->prev_dt = g_param->dt;
        g_param->time +=
            static_cast<double>(g_param->dt) / static_cast<double>(playback);
        result->time = g_param->time;
    } else {
        result->time = 0.0;
    }

    unsigned long long step_ms = parse_step_ms();
    if (step_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    }
}

extern "C" DLL_EXPORT void fetch() {
    mem::copy_from_device_to_host(g_dev_dataset.vertex.curr.data,
                                  g_host_dataset.vertex.curr.data,
                                  g_host_dataset.vertex.curr.size);
    mem::copy_from_device_to_host(g_dev_dataset.vertex.prev.data,
                                  g_host_dataset.vertex.prev.data,
                                  g_host_dataset.vertex.prev.size);
}

extern "C" DLL_EXPORT void fetch_inv_rest() {
    mem::copy_from_device_to_host(g_dev_dataset.inv_rest2x2.data,
                                  g_host_dataset.inv_rest2x2.data,
                                  g_host_dataset.inv_rest2x2.size);
    mem::copy_from_device_to_host(g_dev_dataset.inv_rest3x3.data,
                                  g_host_dataset.inv_rest3x3.data,
                                  g_host_dataset.inv_rest3x3.size);
}

extern "C" DLL_EXPORT void fetch_rest_angles() {
    if (g_host_dataset.prop.hinge.size > 0) {
        mem::copy_from_device_to_host(g_dev_dataset.prop.hinge.data,
                                      g_host_dataset.prop.hinge.data,
                                      g_host_dataset.prop.hinge.size);
    }
    if (g_host_dataset.prop.vertex.size > 0) {
        mem::copy_from_device_to_host(g_dev_dataset.prop.vertex.data,
                                      g_host_dataset.prop.vertex.data,
                                      g_host_dataset.prop.vertex.size);
    }
}

extern "C" DLL_EXPORT void fetch_dyn_counts(unsigned *n_value,
                                            unsigned *n_offset) {
    *n_value = 0;
    *n_offset = 0;
}

extern "C" DLL_EXPORT void fetch_dyn(unsigned * /*index*/, Mat3x3f * /*value*/,
                                     unsigned * /*offset*/) {
    // No contact assembly happens in the emulator; the dynamic CSR
    // is empty so there's nothing to fetch.
}

extern "C" DLL_EXPORT void update_dyn(unsigned * /*index*/,
                                      unsigned * /*offset*/) {
    // Mirrors the empty dyn_hess after fetch_dyn_counts returns 0/0.
}

extern "C" DLL_EXPORT void update_constraint(const Constraint *constraint) {
    g_host_dataset.constraint = *constraint;
    mem::copy_to_device(constraint->fix, g_dev_dataset.constraint.fix);
    mem::copy_to_device(constraint->pull, g_dev_dataset.constraint.pull);
    mem::copy_to_device(constraint->torque_groups,
                        g_dev_dataset.constraint.torque_groups);
    mem::copy_to_device(constraint->torque_vertices,
                        g_dev_dataset.constraint.torque_vertices);
    mem::copy_to_device(constraint->stitch, g_dev_dataset.constraint.stitch);
    mem::copy_to_device(constraint->sphere, g_dev_dataset.constraint.sphere);
    mem::copy_to_device(constraint->floor, g_dev_dataset.constraint.floor);

    // Production CUDA's update_constraint doesn't touch vertex.curr;
    // the kernel inside advance() pulls the kinematic position from
    // dev_dataset.constraint.fix during integration. Since the
    // emulator's advance() is a no-op, we do that pin write here so
    // the next fetch() round-trip reflects the constraint.
    auto &dev_curr = g_dev_dataset.vertex.curr;
    unsigned total = dev_curr.size;
    for (unsigned i = 0; i < constraint->fix.size; ++i) {
        const FixPair &pair = constraint->fix.data[i];
        if (!pair.kinematic) {
            continue;
        }
        if (pair.index >= total) {
            continue;
        }
        dev_curr.data[pair.index] = pair.position;
    }

    // Rebuild VertexProp.fix_index/pull_index and element-fixed flags
    // so subsequent fetch_rest_angles / fetch_inv_rest / save_state
    // reads see a coherent prop state. Mirrors the CUDA path's
    // bookkeeping at cpp/main/main.cu:1102-1153 in spirit, minus the
    // device upload.
    unsigned vertex_count = g_host_dataset.prop.vertex.size;
    auto &host_vprop = g_host_dataset.prop.vertex;
    if (vertex_count > 0) {
        mem::copy_from_device_to_host(g_dev_dataset.prop.vertex.data,
                                      host_vprop.data, vertex_count);
    }
    unsigned hinge_count = g_host_dataset.prop.hinge.size;
    auto &host_hinge = g_host_dataset.prop.hinge;
    if (hinge_count > 0) {
        mem::copy_from_device_to_host(g_dev_dataset.prop.hinge.data,
                                      host_hinge.data, hinge_count);
    }
    for (unsigned i = 0; i < vertex_count; ++i) {
        host_vprop.data[i].fix_index = 0;
        host_vprop.data[i].pull_index = 0;
    }
    for (unsigned i = 0; i < constraint->fix.size; ++i) {
        host_vprop.data[constraint->fix.data[i].index].fix_index = i + 1;
    }
    for (unsigned i = 0; i < constraint->pull.size; ++i) {
        host_vprop.data[constraint->pull.data[i].index].pull_index = i + 1;
    }
    if (vertex_count > 0) {
        mem::copy_to_device(host_vprop, g_dev_dataset.prop.vertex);
    }

    auto is_fixed = [&](unsigned vi) -> bool {
        return host_vprop.data[vi].fix_index > 0;
    };

    auto &mesh = g_host_dataset.mesh;
    auto &face_prop = g_host_dataset.prop.face;
    for (unsigned i = 0; i < face_prop.size; ++i) {
        Vec3u f = mesh.mesh.face.data[i];
        face_prop.data[i].fixed =
            is_fixed(f[0]) && is_fixed(f[1]) && is_fixed(f[2]);
    }
    if (face_prop.size > 0) {
        mem::copy_to_device(face_prop, g_dev_dataset.prop.face);
    }

    auto &edge_prop = g_host_dataset.prop.edge;
    for (unsigned i = 0; i < edge_prop.size; ++i) {
        Vec2u e = mesh.mesh.edge.data[i];
        edge_prop.data[i].fixed = is_fixed(e[0]) && is_fixed(e[1]);
    }
    if (edge_prop.size > 0) {
        mem::copy_to_device(edge_prop, g_dev_dataset.prop.edge);
    }

    auto &tet_prop = g_host_dataset.prop.tet;
    for (unsigned i = 0; i < tet_prop.size; ++i) {
        Vec4u t = mesh.mesh.tet.data[i];
        tet_prop.data[i].fixed = is_fixed(t[0]) && is_fixed(t[1]) &&
                                 is_fixed(t[2]) && is_fixed(t[3]);
    }
    if (tet_prop.size > 0) {
        mem::copy_to_device(tet_prop, g_dev_dataset.prop.tet);
    }

    auto &hinge_prop = g_host_dataset.prop.hinge;
    for (unsigned i = 0; i < hinge_prop.size; ++i) {
        Vec4u h = mesh.mesh.hinge.data[i];
        hinge_prop.data[i].fixed = is_fixed(h[0]) && is_fixed(h[1]) &&
                                   is_fixed(h[2]) && is_fixed(h[3]);
    }
    if (hinge_prop.size > 0) {
        mem::copy_to_device(hinge_prop, g_dev_dataset.prop.hinge);
    }
}

extern "C" DLL_EXPORT void update_rest_shape(const RestShapeUpdate *update) {
    // Mirror the production path: copy the streamed per-frame inverse rest
    // matrices into the (emulated) device dataset. The emulator's advance is
    // a no-op so elasticity is never evaluated, but keeping the copy faithful
    // means tests that round-trip g_dev_dataset see the updated rest shape.
    if (update->inv_rest2x2.size > 0) {
        mem::copy_to_device(update->inv_rest2x2, g_dev_dataset.inv_rest2x2);
    }
    if (update->inv_rest3x3.size > 0) {
        mem::copy_to_device(update->inv_rest3x3, g_dev_dataset.inv_rest3x3);
    }
    // Mirror production: assign the dedicated per-element `rest_excluded` flag
    // (owned by this path, set wholesale each frame). The emulator's advance is
    // a no-op so this has no dynamical effect, but it keeps g_dev_dataset
    // faithful for any test that round-trips it.
    auto &face_prop = g_host_dataset.prop.face;
    if (update->exclude_face.size) {
        for (unsigned i = 0; i < update->exclude_face.size && i < face_prop.size; ++i) {
            face_prop[i].rest_excluded = update->exclude_face[i] != 0;
        }
        mem::copy_to_device(face_prop, g_dev_dataset.prop.face);
    }
    auto &tet_prop = g_host_dataset.prop.tet;
    if (update->exclude_tet.size) {
        for (unsigned i = 0; i < update->exclude_tet.size && i < tet_prop.size; ++i) {
            tet_prop[i].rest_excluded = update->exclude_tet[i] != 0;
        }
        mem::copy_to_device(tet_prop, g_dev_dataset.prop.tet);
    }
}

extern "C" DLL_EXPORT void override_velocity(const unsigned *indices,
                                             unsigned count, float vx,
                                             float vy, float vz,
                                             float dt) {
    // Bias the implicit predictor by setting prev = curr - v*dt, so the
    // emulated elastic step (pd_arap, when PPF_EMULATED_ELASTIC=1) reads the
    // incoming velocity v from (curr - prev)/dt. With the elastic solver
    // disabled this is harmless: advance() then never integrates prev.
    if (count == 0 || dt <= 0.0f) {
        return;
    }
    auto &curr = g_dev_dataset.vertex.curr;
    auto &prev = g_dev_dataset.vertex.prev;
    for (unsigned i = 0; i < count; ++i) {
        unsigned vi = indices[i];
        if (vi >= curr.size) {
            continue;
        }
        const Vec3f c = curr.data[vi];
        Vec3f np;
        np[0] = (static_cast<float>(c[0]) - vx * dt);
        np[1] = (static_cast<float>(c[1]) - vy * dt);
        np[2] = (static_cast<float>(c[2]) - vz * dt);
        prev.data[vi] = np;
    }
}

extern "C" DLL_EXPORT void gather_current_positions(const unsigned *indices,
                                                    unsigned count,
                                                    float *out) {
    // The emulator's "device" buffers are plain host memory (see mem.hpp),
    // so read vertex.curr directly. Mirrors the CUDA gather so the caller's
    // principal-axis solve uses the live (deformed/rotated) positions.
    if (!out || count == 0) {
        return;
    }
    auto &curr = g_dev_dataset.vertex.curr;
    for (unsigned i = 0; i < count; ++i) {
        unsigned vi = indices[i];
        if (vi < curr.size) {
            out[i * 3 + 0] = static_cast<float>(curr.data[vi][0]);
            out[i * 3 + 1] = static_cast<float>(curr.data[vi][1]);
            out[i * 3 + 2] = static_cast<float>(curr.data[vi][2]);
        } else {
            out[i * 3 + 0] = 0.0f;
            out[i * 3 + 1] = 0.0f;
            out[i * 3 + 2] = 0.0f;
        }
    }
}

extern "C" DLL_EXPORT void override_angular_velocity(
    const unsigned *indices, unsigned count, float wx,
    float wy, float wz, float cx, float cy, float cz,
    float dt) {
    // Inject a rigid spin field: prev -= (ω × (curr - c)) * dt, applied on
    // top of any linear override already written to prev this step. Matches
    // the CUDA path so the emulated elastic solver spins the body.
    if (count == 0 || dt <= 0.0f) {
        return;
    }
    auto &curr = g_dev_dataset.vertex.curr;
    auto &prev = g_dev_dataset.vertex.prev;
    for (unsigned i = 0; i < count; ++i) {
        unsigned vi = indices[i];
        if (vi >= curr.size) {
            continue;
        }
        float rx = static_cast<float>(curr.data[vi][0]) - cx;
        float ry = static_cast<float>(curr.data[vi][1]) - cy;
        float rz = static_cast<float>(curr.data[vi][2]) - cz;
        float vwx = wy * rz - wz * ry;
        float vwy = wz * rx - wx * rz;
        float vwz = wx * ry - wy * rx;
        const Vec3f p = prev.data[vi];
        Vec3f np;
        np[0] = (static_cast<float>(p[0]) - vwx * dt);
        np[1] = (static_cast<float>(p[1]) - vwy * dt);
        np[2] = (static_cast<float>(p[2]) - vwz * dt);
        prev.data[vi] = np;
    }
}

extern "C" DLL_EXPORT void init_collision_windows(const unsigned * /*vert_dmap*/,
                                                  unsigned /*vert_count*/,
                                                  const float * /*windows*/,
                                                  const unsigned * /*window_counts*/,
                                                  unsigned /*n_groups*/) {
    // No BVH / contact pipeline in the emulator; nothing to wire up.
}

extern "C" DLL_EXPORT void refresh_collision_active(float /*time*/) {
    // No active-window propagation needed without a contact pipeline.
}

extern "C" DLL_EXPORT unsigned fetch_intersection_records(IntersectionRecord *out,
                                                          unsigned max_count) {
    unsigned cap = max_count;
    if (cap > MAX_INTERSECTION_RECORDS) {
        cap = MAX_INTERSECTION_RECORDS;
    }
    unsigned n = static_cast<unsigned>(g_synthetic_records.size());
    if (n > cap) {
        n = cap;
    }
    if (n > 0) {
        std::memcpy(out, g_synthetic_records.data(),
                    n * sizeof(IntersectionRecord));
        g_synthetic_records.erase(g_synthetic_records.begin(),
                                  g_synthetic_records.begin() + n);
    }
    return n;
}
