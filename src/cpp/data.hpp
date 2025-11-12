// File: data.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
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
template <class T> using Mat3x5 = SMat<T, 3, 5>;
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
using Mat3x5f = Mat3x5<float>;
using Mat3x6f = Mat3x6<float>;
using Mat4x3f = Mat4x3<float>;
using Mat4x4f = Mat4x4<float>;
using Mat3x9f = Mat3x9<float>;
using Mat6x6f = Mat6x6<float>;
using Mat6x9f = Mat6x9<float>;
using Mat9x9f = Mat9x9<float>;
using Mat9x12f = Mat9x12<float>;
using Mat12x12f = Mat12x12<float>;


enum class Model { ARAP, StVK, BaraffWitkin, SNHk };
enum class Barrier { Cubic, Quad, Log };

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

struct VertexParam {
    float ghat;
    float offset;
    float friction;
};

struct EdgeParam {
    float stiffness;
    float bend;
    float ghat;
    float offset;
    float friction;
};

struct FaceParam {
    Model model;
    float mu;
    float lambda;
    float friction;
    float ghat;
    float offset;
    float bend;
    float strainlimit;
    float shrink;
};

struct HingeParam {
    float bend;
    float ghat;
    float offset;
};

struct TetParam {
    Model model;
    float mu;
    float lambda;
};

struct VertexProp {
    float area;
    float volume;
    float mass;
    unsigned fix_index;
    unsigned pull_index;
    unsigned param_index;
};

struct EdgeProp {
    float length;
    float mass;
    bool fixed;
    unsigned param_index;
};

struct FaceProp {
    float area;
    float mass;
    bool fixed;
    unsigned param_index;
};

struct HingeProp {
    float length;
    bool fixed;
    unsigned param_index;
};

struct TetProp {
    float mass;
    float volume;
    bool fixed;
    unsigned param_index;
};

struct PropSet {
    Vec<VertexProp> vertex;
    Vec<EdgeProp> edge;
    Vec<FaceProp> face;
    Vec<HingeProp> hinge;
    Vec<TetProp> tet;
};

struct ParamArrays {
    Vec<VertexParam> vertex;
    Vec<EdgeParam> edge;
    Vec<FaceParam> face;
    Vec<HingeParam> hinge;
    Vec<TetParam> tet;
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

struct FixPair {
    Vec3f position;
    float ghat;
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
};

struct Sphere {
    Vec3f center;
    float ghat;
    float friction;
    float radius;
    bool bowl;
    bool reverse;
    bool kinematic;
};

struct Floor {
    Vec3f ground;
    float ghat;
    float friction;
    Vec3f up;
    bool kinematic;
};

struct CollisionMesh {
    Vec<Vec3f> vertex;
    Vec<Vec3u> face;
    Vec<Vec2u> edge;
    BVH face_bvh;
    BVH edge_bvh;
    struct {
        Vec<VertexProp> vertex;
        Vec<FaceProp> face;
        Vec<EdgeProp> edge;
    } prop;
    struct {
        Vec<VertexParam> vertex;
        Vec<FaceParam> face;
        Vec<EdgeParam> edge;
    } param_arrays;
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
    float air_friction;
    float air_density;
    float constraint_tol;
    float prev_dt;
    float dt;
    float playback;
    unsigned min_newton_steps;
    float target_toi;
    float stitch_stiffness;
    unsigned cg_max_iter;
    float cg_tol;
    float line_search_max_t;
    float ccd_eps;
    float ccd_reduction;
    unsigned ccd_max_iter;
    float max_dx;
    float eiganalysis_eps;
    float friction_eps;
    float isotropic_air_friction;
    Vec3f gravity;
    Vec3f wind;
    Barrier barrier;
    unsigned csrmat_max_nnz;
    unsigned bvh_alloc_factor;
    float fix_xz;
    bool disable_contact;
    bool fitting;
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
    ParamArrays param_arrays;
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

template <unsigned N> struct Proximity {
    SVecu<N> index;
    SVecf<N> value;
};

#endif
