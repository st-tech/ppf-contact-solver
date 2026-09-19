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
// type is generic so `float` and `unsigned` index vectors also work.

#ifndef LINALG_SMAT_HPP
#define LINALG_SMAT_HPP

// THE QUOTED INCLUDE IS UNCONDITIONAL AND THE ANGLE ONE IS NOT, AND THE SPLIT
// IS THE WHOLE POINT. `la_traits.hpp` resolves its own backend split with
// `__METAL_VERSION__` and includes `<metal_stdlib>` on that side, so the shader
// compiler can read it; the run-time assembler neutralizes a quoted include as
// it splices, so naming it here costs the shipped shader nothing and lets an
// offline unit that names this header get the traits without being told to. Only
// `<cmath>` has to stay behind the role marker, because that one the shader
// compiler genuinely cannot serve.
#include "la_traits.hpp"
#ifndef SM_MSL_CONCAT
#include <cmath>
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#define LA_HD __host__ __device__

namespace linalg {

#ifndef SM_SQRT
#define SM_SQRT sqrtf
#define LA_UNDEF_SM_SQRT
#endif
#ifndef SM_ABS
#define SM_ABS fabsf
#define LA_UNDEF_SM_ABS
#endif
// Division, for the one place in this header where a reciprocal decides whether
// a result is bit-comparable across backends: the 2x2 / 3x3 inverse. CUDA's and
// the host's `/` are correctly rounded, so the fallback below is the operator
// itself and the expansion is textually what was there before; MSL's plain `/`
// is not, and its seam spelling is precise::divide, which is in every math
// mode. Everything else here divides with the operator, because no other site
// feeds a quantity a parity gate compares to the bit.
#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define LA_UNDEF_SM_DIV
#endif
// Address space on every pointer/reference this header declares. MSL requires
// one; CUDA and the host have a single address space and expand it to nothing
// (the same seam shared_math.h and the two device prologues already use).
// SMat, VecRef, MatRef, CommaInit and their operands all live in thread space
// (they are stack-local views/values, never a kernel-entry device argument),
// so this one qualifier covers the whole math body. Map is the one exception:
// its raw pointer keeps whatever address space its Ptr template parameter
// deduced (device/thread/threadgroup), so Ptr is never spelled with SM_THREAD.
#ifndef SM_THREAD
#define SM_THREAD
#define LA_UNDEF_SM_THREAD
#endif

// The smallest NORMAL float, 2^-126. Spelled as a literal on purpose: this
// tree's common.hpp #undefs the standard FLT_MIN and redefines it as -1.0e8f,
// a negative sentinel, so reaching for FLT_MIN here would silently compare
// against the wrong thing. normalize() uses this to decide whether squaring
// was safe; see the comment there.
#define LA_MIN_NORMAL_F32 1.17549435e-38f

// ---- per-scalar helpers -----------------------------------------------------
// float uses the intrinsics. abs on any other scalar resolves via ADL to its
// own free abs(). sqrt on a non-float scalar is computed in float32 (convert
// in, sqrtf, convert back) -- never double, and sqrt() is deliberately left
// undefined on those scalar types. The non-template float overloads are an
// exact match and win over the templates.
static LA_HD inline float la_sqrt(float x) { return SM_SQRT(x); }
static LA_HD inline float la_abs(float x) { return SM_ABS(x); }
template <class T> static LA_HD T la_sqrt(SM_THREAD const T &x) {
    return static_cast<T>(SM_SQRT(static_cast<float>(x)));
}
template <class T> static LA_HD T la_abs(SM_THREAD const T &x) { return abs(x); }

template <class T, int R, int C> struct SMat;
template <class T, int N> struct VecRef;
template <class T, int R, int C> struct MatRef;
template <class V, class Ptr> struct Map;

// ---------------------------------------------------------------------------
// Strided vector view (result of .col(), .row(), .head()/.tail()/.segment()).
// Behaves as an lvalue (write-back) AND converts to a materialized vector.
// stride is a runtime int: 1 for columns/segments, R for rows.
// ---------------------------------------------------------------------------
template <class T, int N> struct VecRef {
    SM_THREAD T *p;
    int s;
    LA_HD VecRef(SM_THREAD T *ptr, int stride) : p(ptr), s(stride) {}

    LA_HD SM_THREAD T &operator[](int i) { return p[i * s]; }
    LA_HD SM_THREAD const T &operator[](int i) const { return p[i * s]; }
    LA_HD SM_THREAD T &operator()(int i) { return p[i * s]; }
    LA_HD SM_THREAD const T &operator()(int i) const { return p[i * s]; }

    // materialize
    LA_HD SMat<T, N, 1> eval() const {
        SMat<T, N, 1> r;
        for (int i = 0; i < N; ++i)
            r.m[i] = p[i * s];
        return r;
    }
    LA_HD operator SMat<T, N, 1>() const { return eval(); }

    // write-back assignment
    template <class O> LA_HD SM_THREAD VecRef &operator=(SM_THREAD const O &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = o[i];
        return *this;
    }
    LA_HD SM_THREAD VecRef &operator=(SM_THREAD const VecRef &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = o[i];
        return *this;
    }
    template <class O> LA_HD SM_THREAD VecRef &operator+=(SM_THREAD const O &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = p[i * s] + o[i];
        return *this;
    }
    template <class O> LA_HD SM_THREAD VecRef &operator-=(SM_THREAD const O &o) {
        for (int i = 0; i < N; ++i)
            p[i * s] = p[i * s] - o[i];
        return *this;
    }

    template <class O> LA_HD T dot(SM_THREAD const O &o) const {
        T acc = T(0);
        for (int i = 0; i < N; ++i)
            acc = acc + p[i * s] * o[i];
        return acc;
    }
    template <class O> LA_HD SMat<T, 3, 1> cross(SM_THREAD const O &o) const {
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
        // Guard exact-zero: a zero-length vector stays zero rather than 0/0=NaN.
        // Eigen divides unconditionally but never on an exactly-zero vector; a
        // degenerate mode that our float rounding lands on zero is weighted by
        // ~0 anyway, so leaving it zero matches the physics without poisoning
        // the assembly with NaN.
        //
        // The guard has to be taken on the SQUARED norm, not on the norm, and
        // the two are not the same test. squaredNorm() squares before la_sqrt()
        // takes the root, which halves the usable exponent range: any vector
        // whose largest component is below sqrt(FLT_MIN) ~ 1.0842e-19 has a
        // squared norm at or below the subnormal floor even though the vector
        // itself is twenty decades clear of it. On a backend that flushes
        // subnormals (Metal does, in EVERY math mode) that squared norm reads
        // as exactly 0, the guard below declines to act, and normalize()
        // silently returns the original NON-UNIT vector. Measured: 5 of 8
        // non-zero test vectors came back unnormalized. The comment above
        // justifies the guard by assuming such a vector is weighted by ~0
        // anyway, which is true of a genuinely zero vector and false of this one.
        T sq = squaredNorm();
        if (sq >= T(LA_MIN_NORMAL_F32)) {
            // Fast path, and BIT-IDENTICAL to the previous implementation. It
            // runs exactly when the squaring was safe, i.e. when sq came back a
            // NORMAL float, which is the case for every vector the solver
            // actually handles. Nothing about that arithmetic changes.
            T n = la_sqrt(sq);
            for (int i = 0; i < N; ++i)
                p[i * s] = p[i * s] / n;
            return;
        }
        // sq is zero or subnormal, so the squaring lost information: either
        // every square underflowed to zero, or they landed in the subnormal
        // range where they carry fewer than 24 significant bits and la_sqrt
        // returns a visibly wrong length (measured: 1.0097 instead of 1 for an
        // isotropic vector of component 1e-22 on CUDA). Both are repaired the
        // same way, by dividing through a scale factor so the squares stay in
        // range.
        //
        // The repair works because the COMPONENTS are still normal even when
        // their squares are not, which is the whole asymmetry being exploited.
        // It therefore stops working one binade further down: if the largest
        // component is itself subnormal, then on a flush-to-zero backend
        // p[i] / mx flushes both operands and yields 0/0 = NaN, the `n > 0`
        // test below is false, and the vector comes back unchanged and
        // non-unit. That is not a regression (the pre-fix code did the same),
        // it is not reachable on Metal by arithmetic (a backend that flushes
        // cannot PRODUCE a subnormal, so such a vector would have to be written
        // by the host), and it is not fixable in fp32 under flush-to-zero at
        // all. Recorded so nobody reads the line above as a stronger promise
        // than it is.
        T mx = T(0);
        for (int i = 0; i < N; ++i) {
            T a = p[i * s] < T(0) ? -p[i * s] : p[i * s];
            if (a > mx)
                mx = a;
        }
        if (!(mx > T(0)))
            return; // genuinely zero, or NaN: unchanged from before
        T acc = T(0);
        for (int i = 0; i < N; ++i) {
            T q = p[i * s] / mx;
            acc = acc + q * q;
        }
        T n = la_sqrt(acc); // in [1, sqrt(N)] by construction, cannot underflow
        if (n > T(0))
            for (int i = 0; i < N; ++i)
                p[i * s] = (p[i * s] / mx) / n;
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
    SM_THREAD T *base;
    int ld;
    LA_HD MatRef(SM_THREAD T *b, int leading) : base(b), ld(leading) {}

    LA_HD SM_THREAD T &operator()(int r, int c) { return base[r + c * ld]; }
    LA_HD SM_THREAD const T &operator()(int r, int c) const {
        return base[r + c * ld];
    }

    LA_HD SMat<T, R, C> eval() const {
        SMat<T, R, C> out;
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                out.m[r + c * R] = base[r + c * ld];
        return out;
    }
    LA_HD operator SMat<T, R, C>() const { return eval(); }

    template <class O> LA_HD SM_THREAD MatRef &operator=(SM_THREAD const O &o) {
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                base[r + c * ld] = o(r, c);
        return *this;
    }
    LA_HD SM_THREAD MatRef &operator=(SM_THREAD const MatRef &o) {
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                base[r + c * ld] = o(r, c);
        return *this;
    }
    template <class O> LA_HD SM_THREAD MatRef &operator+=(SM_THREAD const O &o) {
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
                base[r + c * ld] = base[r + c * ld] + o(r, c);
        return *this;
    }
    template <class O> LA_HD SM_THREAD MatRef &operator-=(SM_THREAD const O &o) {
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
    SM_THREAD SMat<T, R, C> *dst;
    int row, col, bandRows;
    LA_HD CommaInit(SM_THREAD SMat<T, R, C> *d)
        : dst(d), row(0), col(0), bandRows(0) {}

    template <int BR, int BC>
    LA_HD void put(SM_THREAD const SMat<T, BR, BC> &b) {
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
    LA_HD void put_scalar(SM_THREAD const T &s) {
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
    LA_HD SM_THREAD CommaInit &operator,(SM_THREAD const SMat<T, BR, BC> &b) {
        put(b);
        return *this;
    }
    template <int NN>
    LA_HD SM_THREAD CommaInit &operator,(SM_THREAD const VecRef<T, NN> &v) {
        put(v.eval());
        return *this;
    }
    template <int BR, int BC>
    LA_HD SM_THREAD CommaInit &operator,(SM_THREAD const MatRef<T, BR, BC> &b) {
        put(b.eval());
        return *this;
    }
    LA_HD SM_THREAD CommaInit &operator,(SM_THREAD const T &s) {
        put_scalar(s);
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Core fixed-size matrix.
// ---------------------------------------------------------------------------
template <class T, int R, int C> struct SMat {
    using Scalar = T;
    // Enumerators, not `static constexpr int`: this header compiles as both CUDA
    // and MSL, and MSL rejects a program-scope constexpr variable (including a
    // static data member) with "program scope variable must reside in constant
    // address space". All three are int, so one enum with int as the fixed
    // underlying type reproduces the previous types exactly.
    enum : int { Rows = R, Cols = C, Size = R * C };

    T m[R * C];

    LA_HD SMat() {}

    // Variadic scalar constructor: Vec3f(x,y,z), Vec2f(a,b), Vec4f(...).
    // Fills linearly (column-major). Enabled only when the argument count
    // matches the element count and is >= 2 (so it never shadows copy/convert).
    template <class... A,
              class = traits::enable_if_t<sizeof...(A) == (unsigned)(R * C) &&
                                          (sizeof...(A) >= 2)>>
    LA_HD SMat(A... args) {
        T tmp[] = {T(args)...};
        for (int i = 0; i < R * C; ++i)
            m[i] = tmp[i];
    }

    // Single-scalar constructor for a 1-element vector: Vec1u(i), Vec1f(x).
    template <class U,
              class = traits::enable_if_t<
                  R * C == 1 && traits::is_convertible<U, T>::value &&
                  !traits::is_same<traits::remove_cvref_t<U>, SMat>::value>>
    LA_HD SMat(U v) {
        m[0] = T(v);
    }

    // Convert from a strided vector view (vector case).
    template <int NN,
              class = traits::enable_if_t<C == 1 && NN == R>>
    LA_HD SMat(SM_THREAD const VecRef<T, NN> &v) {
        for (int i = 0; i < R; ++i)
            m[i] = v[i];
    }

    // element access ---------------------------------------------------------
    LA_HD SM_THREAD T &operator()(int r, int c) { return m[r + c * R]; }
    LA_HD SM_THREAD const T &operator()(int r, int c) const {
        return m[r + c * R];
    }
    LA_HD SM_THREAD T &operator()(int i) { return m[i]; }
    LA_HD SM_THREAD const T &operator()(int i) const { return m[i]; }
    LA_HD SM_THREAD T &operator[](int i) { return m[i]; }
    LA_HD SM_THREAD const T &operator[](int i) const { return m[i]; }
    LA_HD SM_THREAD T *data() { return m; }
    LA_HD SM_THREAD const T *data() const { return m; }
    LA_HD constexpr int rows() const { return R; }
    LA_HD constexpr int cols() const { return C; }
    LA_HD constexpr int size() const { return R * C; }
    LA_HD SMat eval() const { return *this; }

    // in-place arithmetic -----------------------------------------------------
    LA_HD SM_THREAD SMat &operator+=(SM_THREAD const SMat &o) {
        for (int i = 0; i < R * C; ++i)
            m[i] = m[i] + o.m[i];
        return *this;
    }
    LA_HD SM_THREAD SMat &operator-=(SM_THREAD const SMat &o) {
        for (int i = 0; i < R * C; ++i)
            m[i] = m[i] - o.m[i];
        return *this;
    }
    LA_HD SM_THREAD SMat &operator*=(T s) {
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
    template <class O> LA_HD T dot(SM_THREAD const O &o) const {
        T acc = T(0);
        for (int i = 0; i < R * C; ++i)
            acc = acc + m[i] * o[i];
        return acc;
    }
    template <class O> LA_HD SMat<T, 3, 1> cross(SM_THREAD const O &o) const {
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
        // Guard the SQUARED norm, not the norm; see VecRef::normalize for why
        // those are different tests and for the measurement behind it.
        T sq = squaredNorm();
        if (sq >= T(LA_MIN_NORMAL_F32)) {
            // Fast path, bit-identical to the previous implementation, and
            // taken exactly when the squaring was safe. See VecRef::normalize.
            T n = la_sqrt(sq);
            for (int i = 0; i < R * C; ++i)
                m[i] = m[i] / n;
            return;
        }
        T mx = T(0);
        for (int i = 0; i < R * C; ++i) {
            T a = m[i] < T(0) ? -m[i] : m[i];
            if (a > mx)
                mx = a;
        }
        if (!(mx > T(0)))
            return; // genuinely zero, or NaN: unchanged from before
        T acc = T(0);
        for (int i = 0; i < R * C; ++i) {
            T q = m[i] / mx;
            acc = acc + q * q;
        }
        T n = la_sqrt(acc);
        if (n > T(0))
            for (int i = 0; i < R * C; ++i)
                m[i] = (m[i] / mx) / n;
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
    template <class I> LA_HD T minCoeff(SM_THREAD I *idx) const {
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
    template <class I> LA_HD T maxCoeff(SM_THREAD I *idx) const {
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
        return VecRef<T, R>(const_cast<SM_THREAD T *>(&m[c * R]), 1);
    }
    LA_HD VecRef<T, C> row(int r) { return VecRef<T, C>(&m[r], R); }
    LA_HD VecRef<T, C> row(int r) const {
        return VecRef<T, C>(const_cast<SM_THREAD T *>(&m[r]), R);
    }
    template <int BR, int BC> LA_HD MatRef<T, BR, BC> block(int i, int j) {
        return MatRef<T, BR, BC>(&m[i + j * R], R);
    }
    template <int BR, int BC>
    LA_HD MatRef<T, BR, BC> block(int i, int j) const {
        return MatRef<T, BR, BC>(const_cast<SM_THREAD T *>(&m[i + j * R]), R);
    }
    template <int NN> LA_HD VecRef<T, NN> head() {
        return VecRef<T, NN>(&m[0], 1);
    }
    template <int NN> LA_HD VecRef<T, NN> head() const {
        return VecRef<T, NN>(const_cast<SM_THREAD T *>(&m[0]), 1);
    }
    template <int NN> LA_HD VecRef<T, NN> tail() {
        return VecRef<T, NN>(&m[R * C - NN], 1);
    }
    template <int NN> LA_HD VecRef<T, NN> tail() const {
        return VecRef<T, NN>(const_cast<SM_THREAD T *>(&m[R * C - NN]), 1);
    }
    template <int NN> LA_HD VecRef<T, NN> segment(int i) {
        return VecRef<T, NN>(&m[i], 1);
    }
    template <int NN> LA_HD VecRef<T, NN> segment(int i) const {
        return VecRef<T, NN>(const_cast<SM_THREAD T *>(&m[i]), 1);
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
            T inv = SM_DIV(T(1), d);
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
            T inv = SM_DIV(T(1), det);
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
    LA_HD CommaInit<T, R, C> operator<<(SM_THREAD const SMat<T, BR, BC> &b) {
        CommaInit<T, R, C> ci(this);
        ci.put(b);
        return ci;
    }
    template <int NN>
    LA_HD CommaInit<T, R, C> operator<<(SM_THREAD const VecRef<T, NN> &v) {
        CommaInit<T, R, C> ci(this);
        ci.put(v.eval());
        return ci;
    }
    template <int BR, int BC>
    LA_HD CommaInit<T, R, C> operator<<(SM_THREAD const MatRef<T, BR, BC> &b) {
        CommaInit<T, R, C> ci(this);
        ci.put(b.eval());
        return ci;
    }
    LA_HD CommaInit<T, R, C> operator<<(SM_THREAD const T &s) {
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
LA_HD SMat<T, R, C> operator+(SM_THREAD const SMat<T, R, C> &a,
                              SM_THREAD const SMat<T, R, C> &b) {
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] + b.m[i];
    return r;
}
template <class T, int R, int C>
LA_HD SMat<T, R, C> operator-(SM_THREAD const SMat<T, R, C> &a,
                              SM_THREAD const SMat<T, R, C> &b) {
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] - b.m[i];
    return r;
}
template <class T, int R, int C>
LA_HD SMat<T, R, C> operator-(SM_THREAD const SMat<T, R, C> &a) {
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = -a.m[i];
    return r;
}
// matrix * matrix (and matrix * vector when Cc==1)
template <class T, int R, int K, int Cc>
LA_HD SMat<T, R, Cc> operator*(SM_THREAD const SMat<T, R, K> &a,
                               SM_THREAD const SMat<T, K, Cc> &b) {
    SMat<T, R, Cc> r = SMat<T, R, Cc>::Zero();
    for (int c = 0; c < Cc; ++c)
        for (int k = 0; k < K; ++k) {
            T bkc = b.m[k + c * K];
            for (int rr = 0; rr < R; ++rr)
                r.m[rr + c * R] = r.m[rr + c * R] + a.m[rr + k * R] * bkc;
        }
    return r;
}
// Scalar multiply/divide accept any arithmetic (or same-T, e.g. float)
// scalar and convert it to the matrix scalar, matching Eigen. The enable_if
// keeps these from competing with matrix*matrix when the operand is a matrix.
template <class S> struct is_la_scalar {
    // Enumerator with bool as the fixed underlying type, for the reason given on
    // SMat's enum. The trait itself comes from la_traits.hpp, which resolves it
    // to the standard library on CUDA and the host and to metal_stdlib under MSL.
    enum : bool { value = traits::is_arithmetic<S>::value };
};
template <class S, class T, int R, int C,
          class = traits::enable_if_t<is_la_scalar<S>::value ||
                                      traits::is_same<S, T>::value>>
LA_HD SMat<T, R, C> operator*(S s, SM_THREAD const SMat<T, R, C> &a) {
    T ts = T(s);
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = ts * a.m[i];
    return r;
}
template <class S, class T, int R, int C,
          class = traits::enable_if_t<is_la_scalar<S>::value ||
                                      traits::is_same<S, T>::value>>
LA_HD SMat<T, R, C> operator*(SM_THREAD const SMat<T, R, C> &a, S s) {
    T ts = T(s);
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] * ts;
    return r;
}
template <class S, class T, int R, int C,
          class = traits::enable_if_t<is_la_scalar<S>::value ||
                                      traits::is_same<S, T>::value>>
LA_HD SMat<T, R, C> operator/(SM_THREAD const SMat<T, R, C> &a, S s) {
    T ts = T(s);
    SMat<T, R, C> r;
    for (int i = 0; i < R * C; ++i)
        r.m[i] = a.m[i] / ts;
    return r;
}

// scalar * block-view (used as (scalar) * M.block<3,3>(...))
template <class T, int R, int C>
LA_HD SMat<T, R, C> operator*(T s, SM_THREAD const MatRef<T, R, C> &a) {
    return s * a.eval();
}

// arithmetic on strided vector views (deduction cannot see the implicit
// conversion to SMat, so provide the exact combinations used).
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(SM_THREAD const VecRef<T, N> &a) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = -a[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(SM_THREAD const VecRef<T, N> &a,
                              SM_THREAD const VecRef<T, N> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a[i] - b[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(SM_THREAD const SMat<T, N, 1> &a,
                              SM_THREAD const VecRef<T, N> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a.m[i] - b[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator-(SM_THREAD const VecRef<T, N> &a,
                              SM_THREAD const SMat<T, N, 1> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a[i] - b.m[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator+(SM_THREAD const VecRef<T, N> &a,
                              SM_THREAD const VecRef<T, N> &b) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = a[i] + b[i];
    return r;
}
template <class T, int N>
LA_HD SMat<T, N, 1> operator*(T s, SM_THREAD const VecRef<T, N> &a) {
    SMat<T, N, 1> r;
    for (int i = 0; i < N; ++i)
        r.m[i] = s * a[i];
    return r;
}

// ---------------------------------------------------------------------------
// map<V>(pointer): view a raw pointer as a fixed-size vector or matrix. Ptr is a
// template parameter so an MSL device/thread/threadgroup qualifier survives
// deduction instead of being erased by an unqualified Scalar* member.
// ---------------------------------------------------------------------------
template <class V, class Ptr> struct Map {
    using Base = traits::remove_const_t<V>;
    using Scalar = typename Base::Scalar;
    // Enumerator, for the reason given on SMat's enum. Base::Size is itself an
    // enumerator there and converts to int, so it is a valid initializer.
    enum : int { Size = Base::Size };
    Ptr p;
    LA_HD Map(Ptr ptr) : p(ptr) {}

    LA_HD Scalar operator[](int i) const { return p[i]; }
    LA_HD operator Base() const {
        Base r;
        for (int i = 0; i < Size; ++i)
            r.m[i] = p[i];
        return r;
    }
    template <class O, bool Writable = !traits::is_const<V>::value,
              class = traits::enable_if_t<Writable>>
    LA_HD SM_THREAD Map &operator=(SM_THREAD const O &o) {
        for (int i = 0; i < Size; ++i)
            p[i] = o[i];
        return *this;
    }
    template <class O, bool Writable = !traits::is_const<V>::value,
              class = traits::enable_if_t<Writable>>
    LA_HD SM_THREAD Map &operator+=(SM_THREAD const O &o) {
        for (int i = 0; i < Size; ++i)
            p[i] = p[i] + o[i];
        return *this;
    }
    template <class O, bool Writable = !traits::is_const<V>::value,
              class = traits::enable_if_t<Writable>>
    LA_HD SM_THREAD Map &operator-=(SM_THREAD const O &o) {
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

template <class V, class Ptr> LA_HD Map<V, Ptr> map(Ptr p) {
    return Map<V, Ptr>(p);
}

// scalar * Map (deduction cannot see Map's conversion to its value type).
template <class V, class Ptr>
LA_HD typename Map<V, Ptr>::Base operator*(typename Map<V, Ptr>::Scalar s,
                                           SM_THREAD const Map<V, Ptr> &mp) {
    typename Map<V, Ptr>::Base r;
    for (int i = 0; i < Map<V, Ptr>::Size; ++i)
        r.m[i] = s * mp.p[i];
    return r;
}

// A column vector is an Nx1 matrix.
template <class T, int N> using SVec = SMat<T, N, 1>;

} // namespace linalg

#undef LA_HD
#undef LA_MIN_NORMAL_F32
#ifdef LA_UNDEF_SM_SQRT
#undef SM_SQRT
#undef LA_UNDEF_SM_SQRT
#endif
#ifdef LA_UNDEF_SM_ABS
#undef SM_ABS
#undef LA_UNDEF_SM_ABS
#endif
#ifdef LA_UNDEF_SM_DIV
#undef SM_DIV
#undef LA_UNDEF_SM_DIV
#endif
#ifdef LA_UNDEF_SM_THREAD
#undef SM_THREAD
#undef LA_UNDEF_SM_THREAD
#endif
#endif
