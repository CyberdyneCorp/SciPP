#pragma once
// scypp::integrate — port of scipy.integrate (Phase 5 subset): fixed-sample and
// adaptive quadrature, and explicit Runge-Kutta initial-value ODE integration.

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "numpp/core/ndarray.hpp"

namespace scypp::integrate {

using numpp::ndarray;
using Integrand = std::function<double(double)>;             // R → R
using OdeFn = std::function<ndarray(double, const ndarray&)>;  // (t, y) → dy/dt

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

// ---- initial-value ODE integration ----
OdeResult solve_ivp(const OdeFn& f, std::pair<double, double> t_span, const ndarray& y0,
                    const std::string& method = "RK45",
                    std::optional<ndarray> t_eval = std::nullopt, double rtol = 1e-3,
                    double atol = 1e-6);

}  // namespace scypp::integrate
