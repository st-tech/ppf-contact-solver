// File: plasticity.hpp
// License: Apache v2.0

#ifndef PLASTICITY_DEF_HPP
#define PLASTICITY_DEF_HPP

#include "../data.hpp"

namespace plasticity {

void update_face_plasticity(DataSet &data, const ParamSet &param);
void update_tet_plasticity(DataSet &data, const ParamSet &param);
void update_hinge_plasticity(DataSet &data, const ParamSet &param);
void update_rod_bend_plasticity(DataSet &data, const ParamSet &param);

} // namespace plasticity

#endif
