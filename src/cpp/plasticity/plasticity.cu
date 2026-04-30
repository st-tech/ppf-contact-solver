// File: plasticity.cu
// License: Apache v2.0

#include "../energy/model/dihedral_angle.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "plasticity.hpp"

namespace plasticity {

__device__ static float _dead_zone_creep(float delta, float threshold,
                                         float alpha) {
    // Returns the increment to add to rest: α · (Δ − sign(Δ)·threshold).
    // Call site must have already verified |Δ| > threshold.
    float target = (delta > 0.0f) ? threshold : -threshold;
    return alpha * (delta - target);
}


void update_face_plasticity(DataSet &data, const ParamSet &param) {

    unsigned shell_face_count = data.shell_face_count;
    Vec<Vec3f> vertex_curr = data.vertex.curr;
    Vec<Vec3u> face_arr = data.mesh.mesh.face;
    Vec<FaceProp> prop_face = data.prop.face;
    Vec<FaceParam> param_face = data.param_arrays.face;
    Vec<Mat2x2f> inv_rest = data.inv_rest2x2;
    float dt = param.dt;

    DISPATCH_START(shell_face_count)
    [vertex_curr, face_arr, prop_face, param_face, inv_rest,
     dt] __device__(unsigned i) mutable {

        const FaceParam &fparam = param_face[prop_face[i].param_index];
        float plasticity = fparam.plasticity;
        if (plasticity <= 0.0f) {
            return;
        }
        float threshold = fparam.plasticity_threshold;
        float alpha = 1.0f - expf(-plasticity * dt);

        const Vec3u &face = face_arr[i];
        const Vec3f &x0 = vertex_curr[face[0]];
        const Vec3f &x1 = vertex_curr[face[1]];
        const Vec3f &x2 = vertex_curr[face[2]];

        Mat3x2f dx;
        dx.col(0) = x1 - x0;
        dx.col(1) = x2 - x0;

        Mat3x2f F = dx * inv_rest[i];
        Svd3x2 svd = utility::svd3x2(F);

        bool changed = false;
        Vec2f S_new = svd.S;
        for (int k = 0; k < 2; k++) {
            float dev = fabsf(svd.S[k] - 1.0f);
            if (dev > threshold) {
                // Pull toward nearest dead-zone edge
                float target = (svd.S[k] < 1.0f)
                    ? (1.0f - threshold)
                    : (1.0f + threshold);
                S_new[k] = svd.S[k] + alpha * (target - svd.S[k]);
                changed = true;
            }
        }
        if (!changed) {
            return;
        }

        Mat3x2f F_new = svd.U * S_new.asDiagonal() * svd.Vt;

        Vec3f cross = dx.col(0).cross(dx.col(1));
        Vec3f n = cross.normalized();
        Vec3f t = dx.col(0).normalized();
        Vec3f b = n.cross(t).normalized();
        Mat2x3f P1;
        P1.row(0) = t.transpose();
        P1.row(1) = b.transpose();

        Mat2x2f P1_dx = P1 * dx;
        inv_rest[i] = P1_dx.inverse() * (P1 * F_new);
    }
    DISPATCH_END;
}

void update_tet_plasticity(DataSet &data, const ParamSet &param) {

    unsigned tet_count = data.mesh.mesh.tet.size;
    Vec<Vec3f> vertex_curr = data.vertex.curr;
    Vec<Vec4u> tet_arr = data.mesh.mesh.tet;
    Vec<TetProp> prop_tet = data.prop.tet;
    Vec<TetParam> param_tet = data.param_arrays.tet;
    Vec<Mat3x3f> inv_rest = data.inv_rest3x3;
    float dt = param.dt;

    DISPATCH_START(tet_count)
    [vertex_curr, tet_arr, prop_tet, param_tet, inv_rest,
     dt] __device__(unsigned i) mutable {

        const TetParam &tparam = param_tet[prop_tet[i].param_index];
        float plasticity = tparam.plasticity;
        if (plasticity <= 0.0f) {
            return;
        }
        float threshold = tparam.plasticity_threshold;
        float alpha = 1.0f - expf(-plasticity * dt);

        const Vec4u &tet = tet_arr[i];
        const Vec3f &x0 = vertex_curr[tet[0]];
        const Vec3f &x1 = vertex_curr[tet[1]];
        const Vec3f &x2 = vertex_curr[tet[2]];
        const Vec3f &x3 = vertex_curr[tet[3]];

        Mat3x3f dx;
        dx.col(0) = x1 - x0;
        dx.col(1) = x2 - x0;
        dx.col(2) = x3 - x0;

        Mat3x3f F = dx * inv_rest[i];
        Svd3x3 svd = utility::svd3x3_rv(F);

        bool changed = false;
        Vec3f S_new = svd.S;
        for (int k = 0; k < 3; k++) {
            float dev = fabsf(svd.S[k] - 1.0f);
            if (dev > threshold) {
                // Pull toward nearest dead-zone edge
                float target = (svd.S[k] < 1.0f)
                    ? (1.0f - threshold)
                    : (1.0f + threshold);
                S_new[k] = svd.S[k] + alpha * (target - svd.S[k]);
                changed = true;
            }
        }
        if (!changed) {
            return;
        }

        Mat3x3f F_new = svd.U * S_new.asDiagonal() * svd.Vt;
        inv_rest[i] = dx.inverse() * F_new;
    }
    DISPATCH_END;
}

void update_hinge_plasticity(DataSet &data, const ParamSet &param) {

    unsigned hinge_count = data.mesh.mesh.hinge.size;
    if (hinge_count == 0) {
        return;
    }
    Vec<Vec3f> vertex_curr = data.vertex.curr;
    Vec<Vec4u> hinge_arr = data.mesh.mesh.hinge;
    Vec<HingeProp> prop_hinge = data.prop.hinge;
    Vec<HingeParam> param_hinge = data.param_arrays.hinge;
    float dt = param.dt;

    DISPATCH_START(hinge_count)
    [vertex_curr, hinge_arr, prop_hinge, param_hinge,
     dt] __device__(unsigned i) mutable {
        HingeProp &prop = prop_hinge[i];
        if (prop.fixed) {
            return;
        }
        const HingeParam &hparam = param_hinge[prop.param_index];
        float plasticity = hparam.plasticity;
        if (plasticity <= 0.0f) {
            return;
        }
        float threshold = hparam.plasticity_threshold;
        float alpha = 1.0f - expf(-plasticity * dt);

        Vec4u hinge = hinge_arr[i];
        Vec4u remapped = dihedral_angle::remap(hinge);
        const Vec3f &x0 = vertex_curr[remapped[0]];
        const Vec3f &x1 = vertex_curr[remapped[1]];
        const Vec3f &x2 = vertex_curr[remapped[2]];
        const Vec3f &x3 = vertex_curr[remapped[3]];
        float theta = dihedral_angle::face_dihedral_angle(x0, x1, x2, x3);
        float delta = theta - prop.rest_angle;
        if (fabsf(delta) > threshold) {
            prop.rest_angle =
                prop.rest_angle + _dead_zone_creep(delta, threshold, alpha);
        }
    }
    DISPATCH_END;
}

void update_rod_bend_plasticity(DataSet &data, const ParamSet &param) {

    unsigned vert_count = data.vertex.curr.size;
    if (vert_count == 0) {
        return;
    }
    Vec<Vec3f> vertex_curr = data.vertex.curr;
    Vec<VertexProp> prop_vertex = data.prop.vertex;
    Vec<EdgeProp> prop_edge = data.prop.edge;
    Vec<EdgeParam> param_edge = data.param_arrays.edge;
    Vec<Vec2u> edge_arr = data.mesh.mesh.edge;
    float dt = param.dt;

    DISPATCH_START(vert_count)
    [data, vertex_curr, prop_vertex, prop_edge, param_edge, edge_arr,
     dt] __device__(unsigned i) mutable {
        // Match the rod-bend gate in energy.cu: interior rod vertex has
        // exactly 2 edges and 0 faces.
        if (data.mesh.neighbor.vertex.edge.count(i) != 2 ||
            data.mesh.neighbor.vertex.face.count(i) != 0) {
            return;
        }
        unsigned edge_idx_0 = data.mesh.neighbor.vertex.edge(i, 0);
        unsigned edge_idx_1 = data.mesh.neighbor.vertex.edge(i, 1);
        const EdgeParam &ep0 = param_edge[prop_edge[edge_idx_0].param_index];
        const EdgeParam &ep1 = param_edge[prop_edge[edge_idx_1].param_index];
        float plasticity = 0.5f * (ep0.plasticity + ep1.plasticity);
        if (plasticity <= 0.0f) {
            return;
        }
        float threshold =
            0.5f * (ep0.plasticity_threshold + ep1.plasticity_threshold);
        float alpha = 1.0f - expf(-plasticity * dt);

        Vec2u edge_0 = edge_arr[edge_idx_0];
        Vec2u edge_1 = edge_arr[edge_idx_1];
        unsigned j = edge_0[0] == i ? edge_0[1] : edge_0[0];
        unsigned k = edge_1[0] == i ? edge_1[1] : edge_1[0];
        Vec3f e0 = vertex_curr[j] - vertex_curr[i];
        Vec3f e1 = vertex_curr[k] - vertex_curr[i];
        float n0 = e0.norm();
        float n1 = e1.norm();
        if (n0 <= 0.0f || n1 <= 0.0f) {
            return;
        }
        float cos_theta = fmaxf(-1.0f, fminf(1.0f, e0.dot(e1) / (n0 * n1)));
        float theta = acosf(cos_theta);
        VertexProp &vp = prop_vertex[i];
        float delta = theta - vp.rest_bend_angle;
        if (fabsf(delta) > threshold) {
            vp.rest_bend_angle =
                vp.rest_bend_angle + _dead_zone_creep(delta, threshold, alpha);
        }
    }
    DISPATCH_END;
}

} // namespace plasticity
