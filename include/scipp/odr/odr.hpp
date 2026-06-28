#pragma once
// scipp::odr — port of scipy.odr: orthogonal distance regression (ODRPACK-style
// total least squares). A Model holds the fitting function f(beta, x); a Data
// holds the observations x, y with optional per-point standard deviations
// (sx, sy); ODR(data, model, beta0).run() returns the estimated parameters,
// their standard errors, the residual variance, and the sum of squares.
//
// The fit minimizes, over the parameters beta AND the per-point x-offsets delta,
//   sum_i [ (f(beta, x_i + delta_i) - y_i)^2 / sy_i^2 + delta_i^2 / sx_i^2 ],
// i.e. orthogonal (total) least squares. Equal weights (no sx/sy) reduce to
// ordinary TLS / Deming regression.

#include <functional>
#include <optional>
#include <string>

#include "numpp/core/ndarray.hpp"

namespace scipp::odr {

using numpp::ndarray;

// Fitting function: (beta, x) -> y, evaluated elementwise over x.
using ModelFn = std::function<ndarray(const ndarray& beta, const ndarray& x)>;

// Wraps the model function (mirrors scipy.odr.Model).
struct Model {
  ModelFn fcn;
  explicit Model(ModelFn f) : fcn(std::move(f)) {}
};

// Observations with optional per-point standard deviations (mirrors
// scipy.odr.Data, but expressed as sx/sy rather than the raw weights). When sx
// or sy is omitted the corresponding weights are 1 (equal weighting).
struct Data {
  ndarray x;
  ndarray y;
  std::optional<ndarray> sx;  // standard deviation of x (per point or scalar)
  std::optional<ndarray> sy;  // standard deviation of y (per point or scalar)

  Data(ndarray x_, ndarray y_,
       std::optional<ndarray> sx_ = std::nullopt,
       std::optional<ndarray> sy_ = std::nullopt)
      : x(std::move(x_)), y(std::move(y_)), sx(std::move(sx_)), sy(std::move(sy_)) {}
};

// Result of an ODR fit (mirrors scipy.odr.Output).
struct Output {
  ndarray beta;        // estimated parameters
  ndarray sd_beta;     // standard errors of the parameters
  ndarray cov_beta;    // unscaled covariance matrix of the parameters
  ndarray delta;       // estimated x-offsets
  double res_var = 0;  // residual variance = sum_square / (n - p)
  double sum_square = 0;
  bool success = false;
  int nfev = 0;
};

// Orthogonal distance regression driver (mirrors scipy.odr.ODR).
class ODR {
 public:
  ODR(Data data, Model model, ndarray beta0,
      double ftol = 1e-10, double xtol = 1e-10, int maxit = 200)
      : data_(std::move(data)), model_(std::move(model)), beta0_(std::move(beta0)),
        ftol_(ftol), xtol_(xtol), maxit_(maxit) {}

  Output run();

 private:
  Data data_;
  Model model_;
  ndarray beta0_;
  double ftol_;
  double xtol_;
  int maxit_;
};

}  // namespace scipp::odr
