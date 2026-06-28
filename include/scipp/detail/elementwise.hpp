#pragma once
// Lift a scalar kernel over a numpp::ndarray. Element-wise SciPP functions are
// written once as a scalar `double(double)` (optionally closing over fixed
// parameters such as a Bessel order) and mapped here, so no function hand-writes
// an array loop. Inputs are promoted to float64 and made contiguous.

#include <cstdint>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"

namespace scipp::detail {

// Map a unary scalar kernel `f: double -> double` over `x`, returning a new
// float64 array with the same shape.
template <class F>
numpp::ndarray map(const numpp::ndarray& x, F&& f) {
  numpp::ndarray xc = x.astype(numpp::kFloat64).ascontiguousarray();
  numpp::ndarray out(xc.shape(), numpp::kFloat64);
  const double* in = xc.template typed_data<double>();
  double* o = out.template typed_data<double>();
  const int64_t n = xc.size();
  for (int64_t i = 0; i < n; ++i) {
    o[i] = static_cast<double>(f(in[i]));
  }
  return out;
}

}  // namespace scipp::detail
