#pragma once
// scypp::integrate — port of scipy.integrate (Phase 5 subset): fixed-sample and
// adaptive quadrature, and explicit Runge-Kutta initial-value ODE integration.

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scypp::integrate {

using numpp::ndarray;
using Integrand = std::function<double(double)>;             // R → R
using OdeFn = std::function<ndarray(double, const ndarray&)>;  // (t, y) → dy/dt
using Integrand2 = std::function<double(double, double)>;          // f(y, x)
using Integrand3 = std::function<double(double, double, double)>;  // f(z, y, x)
using IntegrandN = std::function<double(const std::vector<double>&)>;
using Bound1 = std::function<double(double)>;          // variable bound g(x)
using Bound2 = std::function<double(double, double)>;  // variable bound h(x, y)
using VecIntegrand = std::function<ndarray(double)>;   // R → Rᵏ (vectorised output)

struct QuadResult {
  double value = 0.0;
  double abserr = 0.0;
};

struct OdeResult {
  ndarray t;        // (n_times,)
  ndarray y;        // (n_states, n_times)
  bool success = false;
  int nfev = 0;
  std::string message;
};

// ---- fixed-sample quadrature ----
double trapezoid(const ndarray& y, std::optional<ndarray> x = std::nullopt, double dx = 1.0);
double simpson(const ndarray& y, std::optional<ndarray> x = std::nullopt, double dx = 1.0);
ndarray cumulative_trapezoid(const ndarray& y, std::optional<ndarray> x = std::nullopt,
                             double dx = 1.0, std::optional<double> initial = std::nullopt);

// ---- adaptive / fixed-order quadrature ----
double fixed_quad(const Integrand& f, double a, double b, int n = 5);
QuadResult quad(const Integrand& f, double a, double b, double epsabs = 1.49e-8,
                double epsrel = 1.49e-8, int limit = 50);

// Romberg integration on [a, b] (Richardson-extrapolated trapezoidal rule).
double romberg(const Integrand& f, double a, double b, double tol = 1.48e-8,
               double rtol = 1.48e-8, int divmax = 10);

// Vector-valued adaptive quadrature: integrate f: R → Rᵏ componentwise.
ndarray quad_vec(const VecIntegrand& f, double a, double b, double epsabs = 1e-8,
                 double epsrel = 1e-8, int limit = 50);

// ---- nested adaptive quadrature ----
// Double integral ∫ₐᵇ ∫_{gfun(x)}^{hfun(x)} f(y, x) dy dx.
QuadResult dblquad(const Integrand2& f, double a, double b, const Bound1& gfun,
                   const Bound1& hfun, double epsabs = 1.49e-8, double epsrel = 1.49e-8);
// Triple integral ∫ₐᵇ ∫_{gfun}^{hfun} ∫_{qfun}^{rfun} f(z, y, x) dz dy dx.
QuadResult tplquad(const Integrand3& f, double a, double b, const Bound1& gfun,
                   const Bound1& hfun, const Bound2& qfun, const Bound2& rfun,
                   double epsabs = 1.49e-8, double epsrel = 1.49e-8);
// N-dimensional integral over a hyper-rectangle with fixed bounds (n ranges).
QuadResult nquad(const IntegrandN& f, const std::vector<std::pair<double, double>>& ranges,
                 double epsabs = 1.49e-8, double epsrel = 1.49e-8);

// ---- initial-value ODE integration ----
OdeResult solve_ivp(const OdeFn& f, std::pair<double, double> t_span, const ndarray& y0,
                    const std::string& method = "RK45",
                    std::optional<ndarray> t_eval = std::nullopt, double rtol = 1e-3,
                    double atol = 1e-6);

}  // namespace scypp::integrate
