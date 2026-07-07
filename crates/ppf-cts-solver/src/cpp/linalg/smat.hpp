// File: smat.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// In-house fixed-size dense linear algebra, a drop-in replacement for the
// subset of Eigen used by the CUDA solver. Header-only, __host__ __device__,
// column-major, POD storage (bare T[R*C], natural alignment == alignof(T),
// matching the nalgebra repr(C) mirror on the Rust side; NOT Eigen's 16-align).
// Eager evaluation (no expression templates): every operator materializes a
// fresh SMat, which is what lets us hand-tune the FLOPs later.
//
// Storage is column-major: element (r,c) lives at m[r + c*R]. A vector is
// SMat<T,N,1>. All arithmetic is float32 on device (never double); the scalar
// type is generic so integer index vectors (e.g. `unsigned`) also work.

#ifndef PPF_LINALG_SMAT_HPP
#define PPF_LINALG_SMAT_HPP

#include <cmath>
#include <initializer_list>
#include <type_traits>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#define LA_HD __host__ __device__

namespace linalg {

// ---- per-scalar helpers -----------------------------------------------------
// float uses the intrinsics. abs on any other scalar resolves via ADL to its
// own free abs(). sqrt on a non-float scalar is computed in float32 (convert
// in, sqrtf, convert back) -- never double, and sqrt() is deliberately left
// undefined on those scalar types. The non-template float overloads are an
// exact match and win over the templates.
static LA_HD inline float la_sqrt(float x) { return sqrtf(x); }
static LA_HD inline float la_abs(float x) { return fabsf(x); }
template <class T> static LA_HD T la_sqrt(const T &x) {
    return static_cast<T>(sqrtf(static_cast<float>(x)));
}
template <class T> static LA_HD T la_abs(const T &x) { return abs(x); }

template <class T, int R, int C> struct SMat;
template <class T, int N> struct VecRef;
template <class T, int R, int C> struct MatRef;
template <class V> struct Map;

// ---------------------------------------------------------------------------
// Strided vector view (result of .col(), .row(), .head()/.tail()/.segment()).
// Behaves as an lvalue (write-back) AND converts to a materialized vector.
// stride is a runtime int: 1 for columns/segments, R for rows.
// ---------------------------------------------------------------------------
template <class T, int N> struct VecRef {
    T *p;
    int s;
    LA_HD VecRef(T *ptr, int stride) : p(ptr), s(stride) {}

    LA_HD T &operator[](int i) { return p[i * s]; }
    LA_HD const T &operator[](int i) const { return p[i * s]; }
    LA_HD T &operator()(int i) { return p[i * s]; }
    LA_HD const T &operator()(int i) const { return p[i * s]; }

    // materialize
    LA_HD SMat<T, N, 1> eval() const {
        SMat<T, N, 1> r;
        for (int i = 0; i < N; ++i)
            r.m[i] = p[i * s];
        return r;
    }
    LA_HD operator SMat<T, N, 1>() const { return eval(); }

    // write-back assignment
    template <class O> LA_HD VecRef &operator=(const O &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = o[i];
        return *this;
    }
    LA_HD VecRef &operator=(const VecRef &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = o[i];
        return *this;
    }
    template <class O> LA_HD VecRef &operator+=(const O &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = p[i * s] + o[i];
        return *this;
    }
    template <class O> LA_HD VecRef &operator-=(const O &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = p[i * s] - o[i];
        return *this;
    }

    template <class O> LA_HD T dot(const O &o) const {
        T acc = T(0);
        for (int i = 0; i < N; ++i)
            acc = acc + p[i * s] * o[i];
        return acc;
    }
    template <class O> LA_HD SMat<T, 3, 1> cross(const O &o) const {
        SMat<T, 3, 1> r;
        r.m[0] = p[1 * s] * o[2] - p[2 * s] * o[1];
        r.m[1] = p[2 * s] * o[0] - p[0 * s] * o[2];
        r.m[2] = p[0 * s] * o[1] - p[1 * s] * o[0];
        return r;
    }
    LA_HD T squaredNorm() const {
        T acc = T(0);
        for (int i = 0; i < N; ++i)
            acc = acc + p[i * s] * p[i * s];
        return acc;
    }
    LA_HD T norm() const { return la_sqrt(squaredNorm()); }
    LA_HD void normalize() {
        T n = norm();
        // Guard exact-zero: a zero-length vector stays zero rather than 0/0=NaN.
        // Eigen divides unconditionally but never on an exactly-zero vector; a
        // degenerate mode that our float rounding lands on zero is weighted by
        // ~0 anyway, so leaving it zero matches the physics without poisoning
        // the assembly with NaN.
        if (n > T(0))
            for (int i = 0; i < N; ++i)
                p[i * s] = p[i * s] / n;
    }
    LA_HD SMat<T, N, 1> normalized() const {
        SMat<T, N, 1> r = eval();
        r.normalize();
        return r;
    }
};

// ---------------------------------------------------------------------------
// Rectangular block view of a parent matrix (result of .block<R,C>(i,j)).
// element (r,c) -> base[r + c*ld], ld = parent row count.
// ---------------------------------------------------------------------------
template <class T, int R, int C> struct MatRef {
    T *base;
    int ld;
    LA_HD MatRef(T *b, int leading) : base(b), ld(leading) {}

    LA_HD T &operator()(int r, int c) { return base[r + c * ld]; }
    LA_HD const T &operator()(int r, int c) const { return base[r + c * ld]; }

    LA_HD SMat<T, R, C> eval() const {
        SMat<T, R, C> out;
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                out.m[r + c * R] = base[r + c * ld];
        return out;
    }
    LA_HD operator SMat<T, R, C>() const { return eval(); }

    template <class O> LA_HD MatRef &operator=(const O &o) {
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                base[r + c * ld] = o(r, c);
        return *this;
    }
    LA_HD MatRef &operator=(const MatRef &o) {
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                base[r + c * ld] = o(r, c);
        return *this;
    }
    template <class O> LA_HD MatRef &operator+=(const O &o) {
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                base[r + c * ld] = base[r + c * ld] + o(r, c);
        return *this;
    }
    template <class O> LA_HD MatRef &operator-=(const O &o) {
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                base[r + c * ld] = base[r + c * ld] - o(r, c);
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Comma initializer: M << a, b, c, ...  Fills ROW-MAJOR with a running (row,col)
// cursor, inserting each operand as a BLOCK (scalar = 1x1, vector = Nx1, matrix
// = its shape). Blocks in a horizontal band must share the same row count.
// Placement is eager; no destructor bookkeeping needed.
// ---------------------------------------------------------------------------
template <class T, int R, int C> struct CommaInit {
    SMat<T, R, C> *dst;
    int row, col, bandRows;
    LA_HD CommaInit(SMat<T, R, C> *d) : dst(d), row(0), col(0), bandRows(0) {}

    template <int BR, int BC> LA_HD void put(const SMat<T, BR, BC> &b) {
        for (int c = 0; c < BC; ++c)
            for (int r = 0; r < BR; ++r)
                (*dst)(row + r, col + c) = b.m[r + c * BR];
        bandRows = BR;
        col += BC;
        if (col >= C) {
            col = 0;
            row += bandRows;
            bandRows = 0;
        }
    }
    LA_HD void put_scalar(const T &s) {
        (*dst)(row, col) = s;
        bandRows = 1;
        col += 1;
        if (col >= C) {
            col = 0;
            row += 1;
            bandRows = 0;
        }
    }

    template <int BR, int BC>
    LA_HD CommaInit &operator,(const SMat<T, BR, BC> &b) {
        put(b);
        return *this;
    }
    template <int NN> LA_HD CommaInit &operator,(const VecRef<T, NN> &v) {
        put(v.eval());
        return *this;
    }
    template <int BR, int BC>
    LA_HD CommaInit &operator,(const MatRef<T, BR, BC> &b) {
        put(b.eval());
        return *this;
    }
    LA_HD CommaInit &operator,(const T &s) {
        put_scalar(s);
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Core fixed-size matrix.
// ---------------------------------------------------------------------------
template <class T, int R, int C> struct SMat {
    using Scalar = T;
    static constexpr int Rows = R;
    static constexpr int Cols = C;
    static constexpr int Size = R * C;

    T m[R * C];

    LA_HD SMat() {}

    // Variadic scalar constructor: Vec3f(x,y,z), Vec2f(a,b), Vec4f(...).
    // Fills linearly (column-major). Enabled only when the argument count
    // matches the element count and is >= 2 (so it never shadows copy/convert).
    template <class... A,
              class = std::enable_if_t<sizeof...(A) == (unsigned)(R * C) &&
                                       (sizeof...(A) >= 2)>>
    LA_HD SMat(A... args) {
        T tmp[] = {T(args)...};
        for (int i = 0; i < R * C; ++i)
            m[i] = tmp[i];
    }

    // Single-scalar constructor for a 1-element vector: Vec1u(i), Vec1f(x).
    template <class U,
              class = std::enable_if_t<
                  R * C == 1 && std::is_convertible<U, T>::value &&
                  !std::is_same<std::decay_t<U>, SMat>::value>>
    LA_HD SMat(U v) {
        m[0] = T(v);
    }

    // Row-major nested-brace constructor: Mat3x2f({{a,b},{c,d},{e,f}}).
    LA_HD SMat(std::initializer_list<std::initializer_list<T>> rows) {
        int r = 0;
        for (const auto &row : rows) {
            int c = 0;
            for (const auto &val : row) {
                m[r + c * R] = val;
                ++c;
            }
            ++r;
        }
    }

    // Convert from a strided vector view (vector case).
    template <int NN,
              class = std::enable_if_t<C == 1 && NN == R>>
    LA_HD SMat(const VecRef<T, NN> &v) {
        for (int i = 0; i < R; ++i)
            m[i] = v[i];
    }

    // element access ---------------------------------------------------------
    LA_HD T &operator()(int r, int c) { return m[r + c * R]; }
    LA_HD const T &operator()(int r, int c) const { return m[r + c * R]; }
    LA_HD T &operator()(int i) { return m[i]; }
    LA_HD const T &operator()(int i) const { return m[i]; }
    LA_HD T &operator[](int i) { return m[i]; }
    LA_HD const T &operator[](int i) const { return m[i]; }
    LA_HD T *data() { return m; }
    LA_HD const T *data() const { return m; }
    LA_HD constexpr int rows() const { return R; }
    LA_HD constexpr int cols() const { return C; }
    LA_HD constexpr int size() const { return R * C; }
    LA_HD SMat eval() const { return *this; }

    // in-place arithmetic -----------------------------------------------------
    LA_HD SMat &operator+=(const SMat &o) {
        for (int i = 0; i < R * C; ++i)
            m[i] = m[i] + o.m[i];
        return *this;
    }
    LA_HD SMat &operator-=(const SMat &o) {
        for (int i = 0; i < R * C; ++i)
            m[i] = m[i] - o.m[i];
        return *this;
    }
    LA_HD SMat &operator*=(T s) {
        for (int i = 0; i < R * C; ++i)
            m[i] = m[i] * s;
        return *this;
    }

    // reductions / vector ops -------------------------------------------------
    LA_HD T sum() const {
        T acc = T(0);
        for (int i = 0; i < R * C; ++i)
            acc = acc + m[i];
        return acc;
    }
    LA_HD T trace() const {
        T acc = T(0);
        int n = R < C ? R : C;
        for (int i = 0; i < n; ++i)
            acc = acc + m[i + i * R];
        return acc;
    }
    template <class O> LA_HD T dot(const O &o) const {
        T acc = T(0);
        for (int i = 0; i < R * C; ++i)
            acc = acc + m[i] * o[i];
        return acc;
    }
    template <class O> LA_HD SMat<T, 3, 1> cross(const O &o) const {
        SMat<T, 3, 1> r;
        r.m[0] = m[1] * o[2] - m[2] * o[1];
        r.m[1] = m[2] * o[0] - m[0] * o[2];
        r.m[2] = m[0] * o[1] - m[1] * o[0];
        return r;
    }
    LA_HD T squaredNorm() const {
        T acc = T(0);
        for (int i = 0; i < R * C; ++i)
            acc = acc + m[i] * m[i];
        return acc;
    }
    LA_HD T norm() const { return la_sqrt(squaredNorm()); }
    LA_HD void normalize() {
        T n = norm();
        if (n > T(0)) // guard exact-zero (see VecRef::normalize)
            for (int i = 0; i < R * C; ++i)
                m[i] = m[i] / n;
    }
    LA_HD SMat normalized() const {
        SMat r = *this;
        r.normalize();
        return r;
    }

    LA_HD T minCoeff() const {
        T best = m[0];
        for (int i = 1; i < R * C; ++i)
            if (m[i] < best)
                best = m[i];
        return best;
    }
    LA_HD T maxCoeff() const {
        T best = m[0];
        for (int i = 1; i < R * C; ++i)
            if (best < m[i])
                best = m[i];
        return best;
    }
    template <class I> LA_HD T minCoeff(I *idx) const {
        T best = m[0];
        int bi = 0;
        for (int i = 1; i < R * C; ++i)
            if (m[i] < best) {
                best = m[i];
                bi = i;
            }
        *idx = (I)bi;
        return best;
    }
    template <class I> LA_HD T maxCoeff(I *idx) const {
        T best = m[0];
        int bi = 0;
        for (int i = 1; i < R * C; ++i)
            if (best < m[i]) {
                best = m[i];
                bi = i;
            }
        *idx = (I)bi;
        return best;
    }

    // transpose ---------------------------------------------------------------
    LA_HD SMat<T, C, R> transpose() const {
        SMat<T, C, R> out;
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                out.m[c + r * C] = m[r + c * R];
        return out;
    }

    // cast --------------------------------------------------------------------
    template <class U> LA_HD SMat<U, R, C> cast() const {
        SMat<U, R, C> out;
        for (int i = 0; i < R * C; ++i)
            out.m[i] = U(m[i]);
        return out;
    }

    // Eigen-compatible isZero: every coeff within dummy precision (1e-5) of 0.
    LA_HD bool isZero(float prec = 1e-5f) const {
        for (int i = 0; i < R * C; ++i)
            if (!(la_abs(m[i]) <= prec))
                return false;
        return true;
    }

    // element-wise abs --------------------------------------------------------
    LA_HD SMat cwiseAbs() const {
        SMat out;
        for (int i = 0; i < R * C; ++i)
            out.m[i] = la_abs(m[i]);
        return out;
    }

    // diagonal (read) ---------------------------------------------------------
    LA_HD SMat<T, (R < C ? R : C), 1> diagonal() const {
        constexpr int N = R < C ? R : C;
        SMat<T, N, 1> out;
        for (int i = 0; i < N; ++i)
            out.m[i] = m[i + i * R];
        return out;
    }
    // asDiagonal: build a dense diagonal matrix from a vector (vector case).
    LA_HD SMat<T, R, R> asDiagonal() const {
        SMat<T, R, R> out = SMat<T, R, R>::Zero();
        for (int i = 0; i < R; ++i)
            out.m[i + i * R] = m[i];
        return out;
    }

    // views -------------------------------------------------------------------
    LA_HD VecRef<T, R> col(int c) { return VecRef<T, R>(&m[c * R], 1); }
    LA_HD VecRef<T, R> col(int c) const {
        return VecRef<T, R>(const_cast<T *>(&m[c * R]), 1);
    }
    LA_HD VecRef<T, C> row(int r) { return VecRef<T, C>(&m[r], R); }
    LA_HD VecRef<T, C> row(int r) const {
        return VecRef<T, C>(const_cast<T *>(&m[r]), R);
    }
    template <int BR, int BC> LA_HD MatRef<T, BR, BC> block(int i, int j) {
        return MatRef<T, BR, BC>(&m[i + j * R], R);
    }
    template <int BR, int BC>
    LA_HD MatRef<T, BR, BC> block(int i, int j) const {
        return MatRef<T, BR, BC>(const_cast<T *>(&m[i + j * R]), R);
    }
    template <int NN> LA_HD VecRef<T, NN> head() {
        return VecRef<T, NN>(&m[0], 1);
    }
    template <int NN> LA_HD VecRef<T, NN> head() const {
        return VecRef<T, NN>(const_cast<T *>(&m[0]), 1);
    }
    template <int NN> LA_HD VecRef<T, NN> tail() {
        return VecRef<T, NN>(&m[R * C - NN], 1);
    }
    template <int NN> LA_HD VecRef<T, NN> tail() const {
        return VecRef<T, NN>(const_cast<T *>(&m[R * C - NN]), 1);
    }
    template <int NN> LA_HD VecRef<T, NN> segment(int i) {
        return VecRef<T, NN>(&m[i], 1);
    }
    template <int NN> LA_HD VecRef<T, NN> segment(int i) const {
        return VecRef<T, NN>(const_cast<T *>(&m[i]), 1);
    }

    // 2x2 / 3x3 determinant + inverse ----------------------------------------
    LA_HD T determinant() const {
        if constexpr (R == 2 && C == 2) {
            return m[0] * m[3] - m[2] * m[1];
        } else if constexpr (R == 3 && C == 3) {
            return m[0] * (m[4] * m[8] - m[7] * m[5]) -
                   m[3] * (m[1] * m[8] - m[7] * m[2]) +
                   m[6] * (m[1] * m[5] - m[4] * m[2]);
        } else {
            return T(0);
        }
    }
    LA_HD SMat inverse() const {
        SMat out;
        if constexpr (R == 2 && C == 2) {
            T d = determinant();
            T inv = T(1) / d;
            out.m[0] = m[3] * inv;
            out.m[1] = -m[1] * inv;
            out.m[2] = -m[2] * inv;
            out.m[3] = m[0] * inv;
        } else if constexpr (R == 3 && C == 3) {
            // cofactor / adjugate, column-major (m[r + c*3]).
            T c00 = m[4] * m[8] - m[7] * m[5];
            T c01 = m[7] * m[2] - m[1] * m[8];
            T c02 = m[1] * m[5] - m[4] * m[2];
            T c10 = m[6] * m[5] - m[3] * m[8];
            T c11 = m[0] * m[8] - m[6] * m[2];
            T c12 = m[3] * m[2] - m[0] * m[5];
            T c20 = m[3] * m[7] - m[6] * m[4];
            T c21 = m[6] * m[1] - m[0] * m[7];
            T c22 = m[0] * m[4] - m[3] * m[1];
            T det = m[0] * c00 + m[3] * c01 + m[6] * c02;
            T inv = T(1) / det;
            // inverse = adjugate^T / det; store column-major.
            out.m[0] = c00 * inv;
            out.m[1] = c01 * inv;
            out.m[2] = c02 * inv;
            out.m[3] = c10 * inv;
            out.m[4] = c11 * inv;
            out.m[5] = c12 * inv;
            out.m[6] = c20 * inv;
            out.m[7] = c21 * inv;
            out.m[8] = c22 * inv;
        }
        return out;
    }

    // comma initializer -------------------------------------------------------
    template <int BR, int BC>
    LA_HD CommaInit<T, R, C> operator<<(const SMat<T, BR, BC> &b) {
        CommaInit<T, R, C> ci(this);
        ci.put(b);
        return ci;
    }
    template <int NN>
    LA_HD CommaInit<T, R, C> operator<<(const VecRef<T, NN> &v) {
        CommaInit<T, R, C> ci(this);
        ci.put(v.eval());
        return ci;
    }
    template <int BR, int BC>
    LA_HD CommaInit<T, R, C> operator<<(const MatRef<T, BR, BC> &b) {
        CommaInit<T, R, C> ci(this);
        ci.put(b.eval());
        return ci;
    }
    LA_HD CommaInit<T, R, C> operator<<(const T &s) {
        CommaInit<T, R, C> ci(this);
        ci.put_scalar(s);
        return ci;
    }

    // static factories --------------------------------------------------------
    static LA_HD SMat Zero() {
        SMat r;
        for (int i = 0; i < R * C; ++i)
            r.m[i] = T(0);
        return r;
    }
    static LA_HD SMat Ones() {
        SMat r;
        for (int i = 0; i < R * C; ++i)
            r.m[i] = T(1);
        return r;
    }
    static LA_HD SMat Constant(T v) {
        SMat r;
        for (int i = 0; i < R * C; ++i)
            r.m[i] = v;
        return r;
    }
    static LA_HD SMat Identity() {
        SMat r = Zero();
        int n = R < C ? R : C;
        for (int i = 0; i < n; ++i)
            r.m[i + i * R] = T(1);
        return r;
    }
    static LA_HD SMat Unit(int i) { // basis vector (vector case)
        SMat r = Zero();
        r.m[i] = T(1);
        return r;
    }
};

// ---- free operators --------------------------------------------------------
template <class T, int R, int C>
LA_HD SMat<T, R, C> operator+(const SMat<T, R, C> &a, const SMat<T, R, C> &b) {
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] + b.m[i];
    return r;
}
template <class T, int R, int C>
LA_HD SMat<T, R, C> operator-(const SMat<T, R, C> &a, const SMat<T, R, C> &b) {
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] - b.m[i];
    return r;
}
template <class T, int R, int C>
LA_HD SMat<T, R, C> operator-(const SMat<T, R, C> &a) {
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = -a.m[i];
    return r;
}
// matrix * matrix (and matrix * vector when Cc==1)
template <class T, int R, int K, int Cc>
LA_HD SMat<T, R, Cc> operator*(const SMat<T, R, K> &a, const SMat<T, K, Cc> &b) {
    SMat<T, R, Cc> r = SMat<T, R, Cc>::Zero();
    for (int c = 0; c < Cc; ++c)
        for (int k = 0; k < K; ++k) {
            T bkc = b.m[k + c * K];
            for (int rr = 0; rr < R; ++rr)
                r.m[rr + c * R] = r.m[rr + c * R] + a.m[rr + k * R] * bkc;
        }
    return r;
}
// Scalar multiply/divide accept any arithmetic (or same-T)
// scalar and convert it to the matrix scalar, matching Eigen. The enable_if
// keeps these from competing with matrix*matrix when the operand is a matrix.
template <class S> struct is_la_scalar {
    static constexpr bool value = std::is_arithmetic<S>::value;
};
template <class S, class T, int R, int C,
          class = std::enable_if_t<is_la_scalar<S>::value ||
                                   std::is_same<S, T>::value>>
LA_HD SMat<T, R, C> operator*(S s, const SMat<T, R, C> &a) {
    T ts = T(s);
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = ts * a.m[i];
    return r;
}
template <class S, class T, int R, int C,
          class = std::enable_if_t<is_la_scalar<S>::value ||
                                   std::is_same<S, T>::value>>
LA_HD SMat<T, R, C> operator*(const SMat<T, R, C> &a, S s) {
    T ts = T(s);
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] * ts;
    return r;
}
template <class S, class T, int R, int C,
          class = std::enable_if_t<is_la_scalar<S>::value ||
                                   std::is_same<S, T>::value>>
LA_HD SMat<T, R, C> operator/(const SMat<T, R, C> &a, S s) {
    T ts = T(s);
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] / ts;
    return r;
}

// scalar * block-view (used as (scalar) * M.block<3,3>(...))
template <class T, int R, int C>
LA_HD SMat<T, R, C> operator*(T s, const MatRef<T, R, C> &a) {
    return s * a.eval();
}

// arithmetic on strided vector views (deduction cannot see the implicit
// conversion to SMat, so provide the exact combinations used).
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(const VecRef<T, N> &a) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = -a[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(const VecRef<T, N> &a, const VecRef<T, N> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a[i] - b[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(const SMat<T, N, 1> &a, const VecRef<T, N> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a.m[i] - b[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(const VecRef<T, N> &a, const SMat<T, N, 1> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a[i] - b.m[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator+(const VecRef<T, N> &a, const VecRef<T, N> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a[i] + b[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator*(T s, const VecRef<T, N> &a) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = s * a[i];
    return r;
}

// ---------------------------------------------------------------------------
// Map<V>: view a raw pointer as a fixed-size vector/matrix V (gather/scatter to
// contiguous float buffers). Const-aware: Map<const Vec9f> reads a const*,
// Map<Vec3f> reads and writes. Materializes to V on read; assigns/accumulates
// element-wise through the pointer on write.
// ---------------------------------------------------------------------------
template <class V> struct Map {
    using Base = std::remove_const_t<V>;
    using Scalar = typename Base::Scalar;
    static constexpr int Size = Base::Size;
    using Ptr =
        std::conditional_t<std::is_const<V>::value, const Scalar *, Scalar *>;
    Ptr p;
    LA_HD Map(Ptr ptr) : p(ptr) {}

    LA_HD Scalar operator[](int i) const { return p[i]; }
    LA_HD operator Base() const {
        Base r;
        for (int i = 0; i < Size; ++i)
            r.m[i] = p[i];
        return r;
    }
    template <class O> LA_HD Map &operator=(const O &o) {
        for (int i = 0; i < Size; ++i)
            p[i] = o[i];
        return *this;
    }
    template <class O> LA_HD Map &operator+=(const O &o) {
        for (int i = 0; i < Size; ++i)
            p[i] = p[i] + o[i];
        return *this;
    }
    template <class O> LA_HD Map &operator-=(const O &o) {
        for (int i = 0; i < Size; ++i)
            p[i] = p[i] - o[i];
        return *this;
    }
    LA_HD Scalar squaredNorm() const {
        Scalar acc = Scalar(0);
        for (int i = 0; i < Size; ++i)
            acc = acc + p[i] * p[i];
        return acc;
    }
    LA_HD Scalar norm() const { return la_sqrt(squaredNorm()); }
};

// scalar * Map (deduction cannot see Map's conversion to its value type).
template <class V>
LA_HD typename Map<V>::Base operator*(typename Map<V>::Scalar s,
                                      const Map<V> &mp) {
    typename Map<V>::Base r;
    for (int i = 0; i < Map<V>::Size; ++i)
        r.m[i] = s * mp.p[i];
    return r;
}

// A column vector is an Nx1 matrix.
template <class T, int N> using SVec = SMat<T, N, 1>;

} // namespace linalg

#undef LA_HD
#endif
