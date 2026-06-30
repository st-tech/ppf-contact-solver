// File: sand.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Phony granular mover for the CUDA-free emulator (libsimbackend_cpu). A
// SAND group decodes to a faceless point cloud of loose vertices (the grain
// positions). This routine just ballistically integrates each free grain
// under gravity with a flat floor, so the SAND round-trip can be exercised
// on a macOS / no-GPU host. It is NOT physics: no inter-grain contact, no
// friction, no packing. It exists ONLY to move the cloud.
//
// It runs AUTOMATICALLY for a point-cloud scene (no faces, no edges, no tets,
// no rods) and is a no-op otherwise, so it needs no env-var opt-in and never
// disturbs SHELL/SOLID/ROD/PDRD scenes (whose free verts are already handled
// by the kinematic / elastic emulator paths).

#ifndef SAND_HPP
#define SAND_HPP

#include "../cpp/data.hpp"

namespace sand {

// Flat floor on the gravity-down (Z) axis. Grains settle here.
constexpr double FLOOR_Z = -2.0;

// True when the dataset is a pure particle cloud: every vertex is a free grain
// referenced by no element (no faces, no derived edges, no tets, no rods).
// This is the inverse of pd_arap's `shell_face_count == 0` gate, but
// element-complete so an edge-only or tet scene is correctly excluded.
inline bool is_point_cloud(const DataSet &dev) {
    return dev.mesh.mesh.face.size == 0
        && dev.mesh.mesh.edge.size == 0
        && dev.mesh.mesh.tet.size == 0
        && dev.rod_count == 0;
}

// One explicit ballistic step per free grain. Mirrors pd_arap's vertex range
// (dev.vertex.curr.size) and its prev/curr write-back, but depends on no
// shell faces. Pinned grains (fix_index > 0) are held fixed. No-op unless the
// scene is a faceless/edgeless point cloud (auto-detected).
inline void step(DataSet &dev, const ParamSet &param) {
    if (!is_point_cloud(dev)) {
        return;
    }
    const int n = static_cast<int>(dev.vertex.curr.size);
    if (n == 0) {
        return;
    }
    const double dt = param.dt;
    if (!(dt > 0.0)) {
        return;
    }
    const double prev_dt = (param.prev_dt > 0.0f) ? param.prev_dt : dt;
    const double gx = param.gravity[0];
    const double gy = param.gravity[1];
    const double gz = param.gravity[2];
    for (int v = 0; v < n; ++v) {
        // Skip pinned grains exactly as the elastic path skips fixed verts.
        if (v < static_cast<int>(dev.prop.vertex.size) &&
            dev.prop.vertex.data[v].fix_index > 0) {
            continue;
        }
        const Vec3f c = dev.vertex.curr.data[v];
        const Vec3f p = dev.vertex.prev.data[v];
        const double cx = float(c[0]), cy = float(c[1]), cz = float(c[2]);
        const double px = float(p[0]), py = float(p[1]), pz = float(p[2]);
        // velocity from finite difference + gravity, then forward step.
        const double vx = (cx - px) / prev_dt + gx * dt;
        const double vy = (cy - py) / prev_dt + gy * dt;
        const double vz = (cz - pz) / prev_dt + gz * dt;
        double nx = cx + vx * dt;
        double ny = cy + vy * dt;
        double nz = cz + vz * dt;
        if (nz < FLOOR_Z) {
            nz = FLOOR_Z;
        }
        // Write back: prev <- start-of-step, curr <- stepped position.
        dev.vertex.prev.data[v] = dev.vertex.curr.data[v];
        Vec3f nc;
        nc[0] = static_cast<float>(nx);
        nc[1] = static_cast<float>(ny);
        nc[2] = static_cast<float>(nz);
        dev.vertex.curr.data[v] = nc;
    }
}

} // namespace sand

#endif
