// License: Apache v2.0

#include "pd_arap.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

namespace {

template <class T> Vec<T> view(T *data, unsigned size) {
    Vec<T> result{};
    result.data = data;
    result.size = size;
    result.allocated = size;
    return result;
}

bool near(double actual, double expected, double tolerance = 1.0e-9) {
    return std::abs(actual - expected) <= tolerance;
}

void test_combined_pd_rows_with_fixed_affine_increment() {
    constexpr int n = 4;
    const double coordinates[n][3] = {
        {0.0, 0.0, 0.0}, {1.7, 0.1, 0.0},
        {0.2, 1.3, 0.4}, {-0.3, 0.2, 1.5}};
    Vec3f initial[n];
    VertexProp props[n]{};
    unsigned lock_index[n] = {0u, 0u, 0u, 0u};
    for (int v = 0; v < n; ++v) {
        initial[v] = Vec3f(static_cast<float>(coordinates[v][0]),
                           static_cast<float>(coordinates[v][1]),
                           static_cast<float>(coordinates[v][2]));
        props[v].mass = static_cast<float>(v + 1);
    }
    TranslationLock lock{};
    lock.axis = Vec3f(0.0f, 0.0f, 1.0f);
    lock.rotation_axis = Vec3f(1.0f, 0.0f, 0.0f);
    lock.rotation_mode = ROTATION_LOCK_ALLOW_ONLY;
    lock.total_mass = 10.0f;
    lock.dmap_index = 9u;

    DataSet data{};
    data.translation_lock = view(&lock, 1u);
    data.translation_lock_index = view(lock_index, n);
    data.translation_lock_initial = view(initial, n);
    data.prop.vertex = view(props, n);

    Eigen::MatrixXd reference(n, 3);
    Eigen::MatrixXd fixed_values(n, 3);
    Eigen::MatrixXd candidate(n, 3);
    for (int v = 0; v < n; ++v) {
        reference.row(v) << coordinates[v][0], coordinates[v][1],
            coordinates[v][2];
        fixed_values.row(v) = reference.row(v);
        candidate.row(v) << 0.4 * (v + 1), -0.7 * (v + 1),
            0.9 * (v + 1);
    }

    // The first vertex is an exact prescribed row. Its nonzero affine motion
    // must enter h - C_fixed p, not be discarded before the Schur solve.
    std::vector<char> is_fixed = {1, 0, 0, 0};
    std::vector<int> reduced = {-1, 0, 1, 2};
    fixed_values.row(0) += Eigen::RowVector3d(0.13, -0.21, 0.17);
    candidate.row(0) = fixed_values.row(0);
    const pd_arap::ConstraintLayout layout{n, 3, is_fixed, reduced};
    const pd_arap::DynamicConstraints constraints =
        pd_arap::build_dynamic_constraints(data, reference, fixed_values,
                                           layout);
    assert(constraints.C.rows() == 4);

    Eigen::VectorXd free_x(9);
    for (int v = 1; v < n; ++v) {
        free_x.segment<3>(3 * reduced[v]) = candidate.row(v).transpose();
    }
    // This standalone regression uses A = I, so A^-1 C^T is C^T.
    pd_arap::project_with_schur(constraints, constraints.C.transpose(), free_x);
    for (int v = 1; v < n; ++v) {
        candidate.row(v) = free_x.segment<3>(3 * reduced[v]).transpose();
    }

    const Eigen::VectorXd residual =
        constraints.C * free_x - constraints.rhs;
    assert(residual.norm() < 1.0e-9);
    pd_arap::check_rotation_tangent(data, reference, candidate, constraints);
    assert(near(pd_arap::translation_lock_residual(data, candidate, 0u).norm(),
                0.0));
}

void test_prohibit_axis_pd_row_preserves_rotation_plane() {
    constexpr int n = 4;
    const double coordinates[n][3] = {
        {0.0, 0.0, 0.0}, {1.7, 0.1, 0.0},
        {0.2, 1.3, 0.4}, {-0.3, 0.2, 1.5}};
    Vec3f initial[n];
    VertexProp props[n]{};
    unsigned lock_index[n] = {0u, 0u, 0u, 0u};
    for (int v = 0; v < n; ++v) {
        initial[v] = Vec3f(static_cast<float>(coordinates[v][0]),
                           static_cast<float>(coordinates[v][1]),
                           static_cast<float>(coordinates[v][2]));
        props[v].mass = static_cast<float>(v + 1);
    }
    TranslationLock lock{};
    lock.axis = Vec3f::Zero();
    lock.rotation_axis = Vec3f(0.0f, 0.0f, 1.0f);
    lock.rotation_mode = ROTATION_LOCK_PROHIBIT_AXIS;
    lock.total_mass = 10.0f;
    lock.dmap_index = 12u;

    DataSet data{};
    data.translation_lock = view(&lock, 1u);
    data.translation_lock_index = view(lock_index, n);
    data.translation_lock_initial = view(initial, n);
    data.prop.vertex = view(props, n);

    Eigen::MatrixXd reference(n, 3);
    Eigen::MatrixXd fixed_values(n, 3);
    for (int v = 0; v < n; ++v) {
        reference.row(v) << coordinates[v][0], coordinates[v][1],
            coordinates[v][2];
        fixed_values.row(v) = reference.row(v);
    }
    const std::vector<char> is_fixed(n, 0);
    const std::vector<int> reduced = {0, 1, 2, 3};
    const pd_arap::ConstraintLayout layout{n, n, is_fixed, reduced};
    const pd_arap::DynamicConstraints constraints =
        pd_arap::build_dynamic_constraints(data, reference, fixed_values,
                                           layout);
    assert(constraints.C.rows() == 1);
    const pd_arap::RotationFrame &frame = constraints.rotation_frames[0];

    Eigen::MatrixXd candidate = reference;
    const Eigen::Vector3d allowed_omega(1.0, -2.0, 0.0);
    for (int v = 0; v < n; ++v) {
        const Eigen::Vector3d r = reference.row(v).transpose() - frame.com;
        candidate.row(v) += allowed_omega.cross(r).transpose();
    }
    Eigen::VectorXd free_x(3 * n);
    for (int v = 0; v < n; ++v) {
        free_x.segment<3>(3 * v) = candidate.row(v).transpose();
    }
    pd_arap::project_with_schur(constraints, constraints.C.transpose(), free_x);
    for (int v = 0; v < n; ++v) {
        candidate.row(v) = free_x.segment<3>(3 * v).transpose();
        const Eigen::Vector3d r =
            reference.row(v).transpose() - frame.com;
        const Eigen::Vector3d expected_step = allowed_omega.cross(r);
        assert((candidate.row(v) - reference.row(v) -
                expected_step.transpose())
                   .norm() < 1.0e-9);
    }
    pd_arap::check_rotation_tangent(data, reference, candidate, constraints);
}

} // namespace

int main() {
    test_combined_pd_rows_with_fixed_affine_increment();
    test_prohibit_axis_pd_row_preserves_rotation_plane();
    std::puts("emulated PD aggregate lock test passed");
    return 0;
}
