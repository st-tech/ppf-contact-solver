// File: la_traits.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The type traits the linalg core needs, named once and resolved per backend.
// One copy of the linalg code compiles as CUDA, as host C++ and as MSL, and the
// three disagree only about where these traits live: CUDA and the host take
// them from <type_traits> in namespace std, and MSL has no C++ standard library
// at all (a bare `#include <type_traits>` is "file not found" there and `std` is
// an undeclared identifier), so it takes them from <metal_stdlib> in namespace
// metal. Call sites spell them `traits::enable_if_t<...>` and never learn which
// backend answered.
//
// THE #ifdef BELOW IS THE ONLY BACKEND TEST IN THIS FILE AND MUST STAY THAT WAY.
// A seam whose conditionals are scattered through it is a fork with extra steps:
// the two halves drift, each acquires a case the other lacks, and the "one copy
// of the device code" property is lost a line at a time. Each branch is a whole
// answer, include and namespace and all, which is why the namespace scaffolding
// is written twice rather than the test being asked twice.
//
// This covers exactly the traits the linalg core uses today. It is not a
// general std shim, and a trait with no call site does not belong here.
//
// ONLY TRAITS metal_stdlib DEFINES UNCONDITIONALLY MAY BE NAMED ON THE MSL
// SIDE. metal_stdlib is not one fixed set of traits: metal_type_traits gates
// `decay`, `common_type`, `add_pointer` and `remove_extent` behind
// `__HAVE_TYPE_TRAIT_COMMON_TYPE__`, `__HAVE_TYPE_TRAIT_ADD_POINTER__` and
// `__HAVE_TYPE_TRAIT_REMOVE_EXTENT__`, which metal_config supplies. Whether one
// of those four exists is therefore a property of the INSTALLED METAL
// TOOLCHAIN, not of the language, and a file naming one compiles on the machine
// it was written on and fails elsewhere with "no type named 'decay' in
// namespace 'metal'". No file in this tree may name those four.
//
// `enable_if`, `conditional`, `remove_const`, `remove_cv`, `remove_reference`
// and the `is_*` predicates carry no such gate, so they are the floor this file
// builds on, and anything else the linalg core wants is composed from them
// below.

#ifndef LINALG_LA_TRAITS_HPP
#define LINALG_LA_TRAITS_HPP

#ifdef __METAL_VERSION__

#include <metal_stdlib>

namespace linalg {
namespace traits {

using metal::conditional_t;
using metal::enable_if_t;
using metal::is_arithmetic;
using metal::is_const;
using metal::is_convertible;
using metal::is_same;
using metal::remove_const_t;
using metal::remove_cv;
using metal::remove_reference;

} // namespace traits
} // namespace linalg

#else

#include <type_traits>

namespace linalg {
namespace traits {

using std::conditional_t;
using std::enable_if_t;
using std::is_arithmetic;
using std::is_const;
using std::is_convertible;
using std::is_same;
using std::remove_const_t;
using std::remove_cv;
using std::remove_reference;

} // namespace traits
} // namespace linalg

#endif

// THE COMPOSED TRAITS, written once for every backend.
//
// This block sits outside the #ifdef because it is not a backend question: it
// is arithmetic on the primitives each branch above already supplied, so one
// definition serves all three compilers and no backend can carry a different
// meaning for it. A copy inside each branch would be two declarations of one
// idea, which is the drift the single test above prevents.
namespace linalg {
namespace traits {

// Strip the reference, then strip the top-level cv qualifiers. This is C++20's
// `std::remove_cvref_t`, written out because the host and CUDA sides compile as
// C++17 and metal_stdlib does not define it.
//
// IT IS NOT `decay`, AND THE DIFFERENCE IS THE ARRAY AND FUNCTION CASES. A full
// decay also converts an array type to a pointer to its element and a function
// type to a pointer to the function, and this converts neither, so
// `remove_cvref_t<int[4]>` is `int[4]` where a decay would give `int *`. The
// name is the narrow one on purpose: a call site that reads `remove_cvref_t`
// gets what it asked for, where a `decay_t` that quietly did less would be read
// as the standard trait and be wrong on the two types it does not handle.
//
// THAT IS ENOUGH FOR EVERY USE IN THIS TREE, and the reason is worth stating
// rather than leaving to be rediscovered. The one call site is the single
// scalar constructor in smat.hpp, which uses the trait to exclude the case
// where `U` is `SMat` itself, so that a copy does not select that constructor
// instead of the copy constructor. Its `U` is deduced from a BY-VALUE
// parameter, and the language has already applied array-to-pointer,
// function-to-pointer and cv-stripping to such a deduction. The trait only has
// to be exact on scalars, class types, and references to them.
//
// A call site that genuinely needs the array or function case cannot be served
// by widening this line, because `add_pointer` and `remove_extent`, the two
// traits a full decay is built from, sit behind the same `__HAVE_TYPE_TRAIT_`
// gates as `decay` itself. It would have to be written here from partial
// specializations, which on the MSL side means one specialization per address
// space: metal_type_traits writes fourteen of them for `remove_reference`
// alone.
template <class T>
using remove_cvref_t =
    typename remove_cv<typename remove_reference<T>::type>::type;

} // namespace traits
} // namespace linalg

#endif // LINALG_LA_TRAITS_HPP
