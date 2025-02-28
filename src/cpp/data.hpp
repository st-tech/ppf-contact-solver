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

template <class T> using Vec1 = SVec<T, 1>;
template <class T> using Vec2 = SVec<T, 2>;
template <class T> using Vec3 = SVec<T, 3>;
template <class T> using Vec4 = SVec<T, 4>;
template <class T> using Vec6 = SVec<T, 6>;
template <class T> using Vec9 = SVec<T, 9>;
template <class T> using Vec12 = SVec<T, 12>;

using Vec1f = Vec1<float>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Vec6f = Vec6<float>;
using Vec9f = Vec9<float>;
using Vec12f = Vec12<float>;

using Vec1u = Vec1<unsigned>;
using Vec2u = Vec2<unsigned>;
using Vec3u = Vec3<unsigned>;
using Vec4u = Vec4<unsigned>;

template <class T, unsigned R, unsigned C>
using SMat = Eigen::Matrix<T, R, C, Eigen::ColMajor>;
template <unsigned R, unsigned C>
using SMatf = Eigen::Matrix<float, R, C, Eigen::ColMajor>;

template <class T> using Mat3x2 = SMat<T, 3, 2>;
template <class T> using Mat2x2 = SMat<T, 2, 2>;
template <class T> using Mat2x3 = SMat<T, 2, 3>;
template <class T> using Mat3x3 = SMat<T, 3, 3>;
template <class T> using Mat3x4 = SMat<T, 3, 4>;
template <class T> using Mat3x6 = SMat<T, 3, 6>;
template <class T> using Mat4x3 = SMat<T, 4, 3>;
template <class T> using Mat4x4 = SMat<T, 4, 4>;
template <class T> using Mat3x9 = SMat<T, 3, 9>;
template <class T> using Mat6x6 = SMat<T, 6, 6>;
template <class T> using Mat6x9 = SMat<T, 6, 9>;
template <class T> using Mat9x9 = SMat<T, 9, 9>;
template <class T> using Mat9x12 = SMat<T, 9, 12>;
template <class T> using Mat12x12 = SMat<T, 12, 12>;

using Mat2x3f = Mat2x3<float>;
using Mat3x2f = Mat3x2<float>;
using Mat2x2f = Mat2x2<float>;
using Mat3x3f = Mat3x3<float>;
using Mat3x4f = Mat3x4<float>;
using Mat3x6f = Mat3x6<float>;
using Mat4x3f = Mat4x3<float>;
using Mat4x4f = Mat4x4<float>;
using Mat3x9f = Mat3x9<float>;
using Mat6x6f = Mat6x6<float>;
using Mat6x9f = Mat6x9<float>;
using Mat9x9f = Mat9x9<float>;
using Mat9x12f = Mat9x12<float>;
using Mat12x12f = Mat12x12<float>;

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
    float strain_limit_eps;
    float contact_ghat;
    float contact_offset;
    float rod_offset;
    float constraint_ghat;
    float constraint_tol;
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
    float line_search_max_t;
    float ccd_tol;
    float ccd_reduction;
    float max_search_dir_vel;
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
