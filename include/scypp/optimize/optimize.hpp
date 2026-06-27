#pragma once
// scypp::optimize — port of scipy.optimize (Phase 4 subset): scalar root finding,
// scalar/multivariate minimization, and nonlinear least squares. Callables are
// std::function; vectors are numpp::ndarray.

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "numpp/core/ndarray.hpp"

namespace scypp::optimize {

using numpp::ndarray;
using ScalarFn = std::function<double(double)>;
using ObjFn = std::function<double(const ndarray&)>;            // Rⁿ → R
using VecFn = std::function<ndarray(const ndarray&)>;           // Rⁿ → Rᵐ
using ModelFn = std::function<ndarray(const ndarray&, const ndarray&)>;  // (x, params) → y

struct OptimizeResult {
  ndarray x;
  double fun = 0.0;
  bool success = false;
  int nit = 0;
  int nfev = 0;
  std::string message;
};

struct RootResult {
  double root = 0.0;
  bool converged = false;
  int iterations = 0;
  int function_calls = 0;
};

struct ScalarMinResult {
  double x = 0.0;
  double fun = 0.0;
  bool success = false;
  int nit = 0;
};

struct LeastSquaresResult {
  ndarray x;
  ndarray fun;
  double cost = 0.0;
  bool success = false;
  int nfev = 0;
};

struct CurveFitResult {
  ndarray popt;
  ndarray pcov;
};

// ---- scalar root finding ----
double brentq(const ScalarFn& f, double a, double b, double xtol = 2e-12,
              double rtol = 8.881784197001252e-16, int maxiter = 100);
double bisect(const ScalarFn& f, double a, double b, double xtol = 2e-12,
              double rtol = 8.881784197001252e-16, int maxiter = 100);
double newton(const ScalarFn& f, double x0, const ScalarFn& fprime = nullptr,
              double tol = 1.48e-8, int maxiter = 50);

// ---- scalar minimization ----
ScalarMinResult minimize_scalar(const ScalarFn& f, const std::string& method = "brent",
                                std::optional<std::pair<double, double>> bounds = std::nullopt,
                                double xtol = 1.48e-8, int maxiter = 500);

// ---- multivariate minimization ----
OptimizeResult minimize(const ObjFn& f, const ndarray& x0, const std::string& method = "BFGS",
                        double tol = 1e-8, int maxiter = 1000);

// ---- least squares / roots ----
LeastSquaresResult least_squares(const VecFn& residual, const ndarray& x0,
                                 double ftol = 1e-8, double xtol = 1e-8, int max_nfev = 1000);
CurveFitResult curve_fit(const ModelFn& model, const ndarray& xdata, const ndarray& ydata,
                         const ndarray& p0, double ftol = 1e-8, double xtol = 1e-8);
ndarray fsolve(const VecFn& F, const ndarray& x0, double xtol = 1.49012e-8, int maxiter = 200);

}  // namespace scypp::optimize
