// File: data.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DATA_HPP
#define DATA_HPP

#include "vec/vec.hpp"
#include <Eigen/Dense>

using Eigen::Map;
template <class T, unsigned N> using SVec = Eigen::Vector<T, N>;
template <unsigned N> using SVecf = SVec<float, N>;
template <unsigned N> using SVecu = SVec<unsigned, N>;

using Vec2f = SVecf<2>;
using Vec3f = SVecf<3>;
using Vec4f = SVecf<4>;
using Vec6f = SVecf<6>;
using Vec9f = SVecf<9>;
using Vec12f = SVecf<12>;
using Vec12d = SVecf<12>;
using Vec1u = SVecu<1>;
using Vec2u = SVecu<2>;
using Vec3u = SVecu<3>;
using Vec4u = SVecu<4>;

template <class T, unsigned R, unsigned C>
using SMat = Eigen::Matrix<T, R, C, Eigen::ColMajor>;
template <unsigned R, unsigned C>
using SMatf = Eigen::Matrix<float, R, C, Eigen::ColMajor>;

using Mat2x3f = SMatf<2, 3>;
using Mat3x2f = SMatf<3, 2>;
using Mat2x2f = SMatf<2, 2>;
using Mat3x3f = SMatf<3, 3>;
using Mat3x4f = SMatf<3, 4>;
using Mat3x6f = SMatf<3, 6>;
using Mat4x3f = SMatf<4, 3>;
using Mat4x4f = SMatf<4, 4>;
using Mat3x9f = SMatf<3, 9>;
using Mat6x6f = SMatf<6, 6>;
using Mat6x9f = SMatf<6, 9>;
using Mat9x9f = SMatf<9, 9>;
using Mat9x12f = SMatf<9, 12>;
using Mat12x12f = SMatf<12, 12>;

struct VertexNeighbor {
    VecVec<unsigned> face;
    VecVec<unsigned> hinge;
    VecVec<unsigned> edge;
    VecVec<unsigned> rod;
};

struct HingeNeighbor {
    VecVec<unsigned> face;
};

struct EdgeNeighbor {
    VecVec<unsigned> face;
};

struct MeshInfo {
    struct {
        Vec<Vec3u> face;
        Vec<Vec4u> hinge;
        Vec<Vec2u> edge;
        Vec<Vec4u> tet;
    } mesh;
    struct {
        VertexNeighbor vertex;
        HingeNeighbor hinge;
        EdgeNeighbor edge;
    } neighbor;
    struct {
        Vec<char> face;
        Vec<char> vertex;
        Vec<char> hinge;
    } type;
};

struct VertexProp {
    float area;
    float volume;
    float mass;
};

struct RodProp {
    float length;
    float radius;
    float mass;
    float stiffness;
};

struct FaceProp {
    float area;
    float mass;
    float mu;
    float lambda;
};

struct HingeProp {
    float length;
};

struct TetProp {
    float mass;
    float volume;
    float mu;
    float lambda;
};

struct PropSet {
    Vec<VertexProp> vertex;
    Vec<RodProp> rod;
    Vec<FaceProp> face;
    Vec<HingeProp> hinge;
    Vec<TetProp> tet;
};

struct BVH {
    Vec<Vec2u> node;
    VecVec<unsigned> level;
};

struct BVHSet {
    BVH face;
    BVH edge;
    BVH vertex;
};

template <unsigned R, unsigned C> struct Svd {
    SMatf<R, C> U;
    SVecf<C> S;
    SMatf<C, C> Vt;
};

using Svd3x2 = Svd<3, 2>;
using Svd3x3 = Svd<3, 3>;

template <unsigned N> struct DiffTable {
    SVecf<N> deda;
    SMatf<N, N> d2ed2a;
};

using DiffTable2 = DiffTable<2>;
using DiffTable3 = DiffTable<3>;

enum class Model { ARAP, StVK, BaraffWitkin, SNHk };
enum class Barrier { Cubic, Quad, Log };

struct FixPair {
    Vec3f position;
    unsigned index;
    bool kinematic;
};

struct PullPair {
    Vec3f position;
    float weight;
    unsigned index;
};

struct Stitch {
    Vec3u index;
    float weight;
    bool active;
};

struct Sphere {
    Vec3f center;
    float radius;
    bool bowl;
    bool reverse;
    bool kinematic;
};

struct Floor {
    Vec3f ground;
    Vec3f up;
    bool kinematic;
};

struct CollisionMesh {
    bool active;
    Vec<Vec3f> vertex;
    Vec<Vec3u> face;
    Vec<Vec2u> edge;
    BVH face_bvh;
    BVH edge_bvh;
    struct {
        VertexNeighbor vertex;
        HingeNeighbor hinge;
        EdgeNeighbor edge;
    } neighbor;
};

struct Constraint {
    Vec<FixPair> fix;
    Vec<PullPair> pull;
    Vec<Sphere> sphere;
    Vec<Floor> floor;
    Vec<Stitch> stitch;
    CollisionMesh mesh;
};

struct ParamSet {
    double time;
    bool fitting;
    float air_friction;
    float air_density;
    float strain_limit_tau;
    float strain_limit_eps;
    float dt_decrease_factor;
    float contact_ghat;
    float contact_offset;
    float rod_offset;
    float constraint_ghat;
    float prev_dt;
    float dt;
    float playback;
    unsigned min_newton_steps;
    float target_toi;
    float bend;
    float rod_bend;
    float stitch_stiffness;
    unsigned cg_max_iter;
    float cg_tol;
    bool enable_retry;
    float line_search_max_t;
    float ccd_tol;
    float ccd_reduction;
    unsigned ccd_max_iters;
    float eiganalysis_eps;
    float friction;
    float friction_eps;
    float isotropic_air_friction;
    Vec3f gravity;
    Vec3f wind;
    Model model_shell;
    Model model_tet;
    Barrier barrier;
    unsigned csrmat_max_nnz;
    unsigned bvh_alloc_factor;
    float fix_xz;
};

struct StepResult {
    double time;
    bool ccd_success;
    bool pcg_success;
    unsigned retry_count;
    bool intersection_free;
    bool success() const {
        return ccd_success && pcg_success && intersection_free;
    }
};

struct VertexSet {
    Vec<Vec3f> prev;
    Vec<Vec3f> curr;
};

struct DataSet {
    VertexSet vertex;
    MeshInfo mesh;
    PropSet prop;
    Vec<Mat2x2f> inv_rest2x2;
    Vec<Mat3x3f> inv_rest3x3;
    Constraint constraint;
    BVHSet bvh;
    VecVec<unsigned> fixed_index_table;
    VecVec<Vec2u> transpose_table;
    unsigned rod_count;
    unsigned shell_face_count;
    unsigned surface_vert_count;
};

/********** CUSTOM TYPES **********/

struct AABB {
    Vec3f min;
    Vec3f max;
};

struct VertexKinematic {
    Vec3f position;
    bool kinematic;
    bool active;
    bool rod;
};

struct Kinematic {
    Vec<VertexKinematic> vertex;
    Vec<bool> face;
    Vec<bool> edge;
    Vec<bool> hinge;
    Vec<bool> tet;
};

template <unsigned N> struct Proximity {
    SVecu<N> index;
    SVecf<N> value;
};

#endif
