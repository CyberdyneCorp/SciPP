#pragma once
// scypp::interpolate — port of scipy.interpolate (Phase 6 subset). Interpolators
// are stateful callables: construct from samples, then evaluate at query points.

#include <optional>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scypp::interpolate {

using numpp::ndarray;

// 1-D interpolation: linear / nearest / previous / next.
class Interp1d {
 public:
  Interp1d(const ndarray& x, const ndarray& y, std::string kind = "linear",
           std::optional<double> fill_value = std::nullopt);
  ndarray operator()(const ndarray& xq) const;
  double operator()(double xq) const;

 private:
  std::vector<double> x_, y_;
  std::string kind_;
  std::optional<double> fill_;
};

// Shared piecewise-cubic backbone (breakpoints + 4 coeffs per interval).
class PiecewiseCubic {
 public:
  ndarray operator()(const ndarray& xq, int nu = 0) const;
  double operator()(double xq, int nu = 0) const;

 protected:
  std::vector<double> x_;   // n breakpoints
  std::vector<double> c_;   // 4*(n-1): a + b t + c t^2 + d t^3, t = xq - x[i]
  double eval_scalar(double xq, int nu) const;
};

class CubicSpline : public PiecewiseCubic {
 public:
  CubicSpline(const ndarray& x, const ndarray& y, std::string bc_type = "not-a-knot");
};

class PchipInterpolator : public PiecewiseCubic {
 public:
  PchipInterpolator(const ndarray& x, const ndarray& y);
};

class Akima1DInterpolator : public PiecewiseCubic {
 public:
  Akima1DInterpolator(const ndarray& x, const ndarray& y);
};

// N-D rectilinear-grid interpolation.
class RegularGridInterpolator {
 public:
  RegularGridInterpolator(std::vector<ndarray> points, const ndarray& values,
                          std::string method = "linear");
  ndarray operator()(const ndarray& xi) const;  // xi shape (m, ndim)

 private:
  std::vector<std::vector<double>> grids_;
  std::vector<double> values_;
  std::vector<int64_t> shape_;
  std::string method_;
};

ndarray interpn(std::vector<ndarray> points, const ndarray& values, const ndarray& xi,
                std::string method = "linear");

// Scattered radial-basis-function interpolation.
class RBFInterpolator {
 public:
  RBFInterpolator(const ndarray& y, const ndarray& d, std::string kernel = "thin_plate_spline",
                  double epsilon = 1.0, int degree = -1);
  ndarray operator()(const ndarray& x) const;  // x shape (m, dim)

 private:
  std::vector<double> y_;      // n*dim centers (row-major)
  std::vector<double> w_;      // n kernel weights
  std::vector<double> poly_;   // npoly polynomial coeffs
  int n_ = 0, dim_ = 0, degree_ = 0, npoly_ = 0;
  std::string kernel_;
  double epsilon_ = 1.0;
};

}  // namespace scypp::interpolate
