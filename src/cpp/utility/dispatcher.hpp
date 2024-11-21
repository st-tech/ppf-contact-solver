// File: dispatcher.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DISPATCHER_HPP
#define DISPATCHER_HPP

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#define DISPATCH_START(n)                                                      \
    {                                                                          \
        thrust::counting_iterator<unsigned> first(0), last(n);                 \
        thrust::for_each(first, last,
#define DISPATCH_END    );                                                     \
    }

#endif