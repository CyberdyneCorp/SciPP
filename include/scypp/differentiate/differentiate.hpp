#pragma once
// scypp::differentiate — port of scipy.differentiate (Phase 5 subset):
// finite-difference derivative, jacobian and hessian.

#include <functional>

#include "numpp/core/ndarray.hpp"

namespace scypp::differentiate {

using numpp::ndarray;
using ScalarFn = std::function<double(double)>;
using ObjFn = std::function<double(const ndarray&)>;     // Rⁿ → R
using VecFn = std::function<ndarray(const ndarray&)>;    // Rⁿ → Rᵐ

struct DerivativeResult {
  double df = 0.0;
  double error = 0.0;
  bool success = false;
};

DerivativeResult derivative(const ScalarFn& f, double x, double initial_step = 0.5);
ndarray jacobian(const VecFn& F, const ndarray& x);
ndarray hessian(const ObjFn& f, const ndarray& x);

}  // namespace scypp::differentiate
