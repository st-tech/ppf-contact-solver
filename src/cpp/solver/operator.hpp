// File: operator.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "../math/vec.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"

class Operators {
  public:
    virtual void apply(const Vec<float> &, Vec<float> &) const = 0;
    virtual void precond(const Vec<float> &, Vec<float> &) const = 0;
    virtual float norm(const Vec<float> &r, Vec<float> &tmp) const {
        DISPATCH_START(r.size)
        [r, tmp] __device__(unsigned i) mutable {
            tmp[i] = fabs(r[i]);
        } DISPATCH_END;
        return utility::sum_array(tmp, r.size);
    }
};

#endif
