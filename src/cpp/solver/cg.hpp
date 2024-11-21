// File: cg.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../data.hpp"
#include "operator.hpp"
#include <cmath>
#include <string>
#include <tuple>

namespace cg {

std::tuple<bool, unsigned, float> solve(const Operators &op, Vec<float> &r,
                                        Vec<float> &x, unsigned max_iter,
                                        float tol) {
    static Vec<float> tmp = Vec<float>::alloc(x.size);
    static Vec<float> z = Vec<float>::alloc(x.size);
    static Vec<float> p = Vec<float>::alloc(x.size);
    static Vec<float> r0 = Vec<float>::alloc(x.size);

    op.apply(x, tmp);
    r.add_scaled(tmp, -1.0f);
    r0.copy(r);
    op.precond(r, z);
    p.copy(z);

    unsigned iter = 1;
    double rz0 = r.inner_product(z);
    double err0 = op.norm(r, tmp);
    if (!err0) {
        return {true, iter, 0.0f};
    } else {
        while (true) {
            op.apply(p, tmp);
            double alpha = rz0 / (double)p.inner_product(tmp);
            x.add_scaled(p, alpha);
            r.add_scaled(tmp, -alpha);
            double err = op.norm(r, tmp);
            double reresid = err / err0;
            if (reresid < tol) {
                return {true, iter, reresid};
            } else if (iter >= max_iter) {
                return {false, iter, reresid};
            }
            op.precond(r, z);
            double rz1 = r.inner_product(z);
            double beta = rz1 / rz0;
            p.combine(z, p, 1.0f, beta);
            rz0 = rz1;
            iter++;
        }
    }
}

} // namespace cg
