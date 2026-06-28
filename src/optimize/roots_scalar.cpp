// Scalar root finders: brentq (Brent's method), bisect, newton (Newton + secant).
#include "scipp/optimize/optimize.hpp"

#include <cmath>

#include "scipp/error.hpp"

namespace scipp::optimize {

double bisect(const ScalarFn& f, double a, double b, double xtol, double rtol, int maxiter) {
  double fa = f(a), fb = f(b);
  if (fa == 0.0) return a;
  if (fb == 0.0) return b;
  if (fa * fb > 0.0) throw scipp::value_error("bisect: f(a) and f(b) must have opposite signs");
  for (int i = 0; i < maxiter; ++i) {
    double m = 0.5 * (a + b);
    double fm = f(m);
    if (fm == 0.0 || 0.5 * (b - a) < xtol + rtol * std::fabs(m)) return m;
    if (fa * fm < 0.0) { b = m; fb = fm; }
    else { a = m; fa = fm; }
  }
  return 0.5 * (a + b);
}

// Brent's method (ports scipy's brentq control flow).
double brentq(const ScalarFn& f, double xa, double xb, double xtol, double rtol, int maxiter) {
  double xpre = xa, xcur = xb;
  double fpre = f(xpre), fcur = f(xcur);
  if (fpre == 0.0) return xpre;
  if (fcur == 0.0) return xcur;
  if (fpre * fcur > 0.0) throw scipp::value_error("brentq: f(a) and f(b) must have opposite signs");

  double xblk = 0.0, fblk = 0.0, spre = 0.0, scur = 0.0;
  for (int i = 0; i < maxiter; ++i) {
    if (fpre * fcur < 0.0) { xblk = xpre; fblk = fpre; spre = scur = xcur - xpre; }
    if (std::fabs(fblk) < std::fabs(fcur)) {
      xpre = xcur; xcur = xblk; xblk = xpre;
      fpre = fcur; fcur = fblk; fblk = fpre;
    }
    double delta = (xtol + rtol * std::fabs(xcur)) / 2.0;
    double sbis = (xblk - xcur) / 2.0;
    if (fcur == 0.0 || std::fabs(sbis) < delta) return xcur;

    if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
      double stry;
      if (xpre == xblk) {  // secant
        stry = -fcur * (xcur - xpre) / (fcur - fpre);
      } else {  // inverse quadratic interpolation
        double dpre = (fpre - fcur) / (xpre - xcur);
        double dblk = (fblk - fcur) / (xblk - xcur);
        stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
      }
      if (2.0 * std::fabs(stry) < std::min(std::fabs(spre), 3.0 * std::fabs(sbis) - delta)) {
        spre = scur; scur = stry;
      } else {
        spre = sbis; scur = sbis;
      }
    } else {
      spre = sbis; scur = sbis;
    }
    xpre = xcur; fpre = fcur;
    if (std::fabs(scur) > delta) xcur += scur;
    else xcur += (sbis > 0.0 ? delta : -delta);
    fcur = f(xcur);
  }
  return xcur;
}

double newton(const ScalarFn& f, double x0, const ScalarFn& fprime, double tol, int maxiter) {
  if (fprime) {
    double x = x0;
    for (int i = 0; i < maxiter; ++i) {
      double fx = f(x);
      if (fx == 0.0) return x;
      double dfx = fprime(x);
      if (dfx == 0.0) break;
      double dx = fx / dfx;
      x -= dx;
      if (std::fabs(dx) < tol) return x;
    }
    return x;
  }
  // Secant method (SciPy's derivative-free newton).
  double p0 = x0;
  double p1 = x0 * (1.0 + 1e-4) + (x0 >= 0.0 ? 1e-4 : -1e-4);
  double q0 = f(p0), q1 = f(p1);
  for (int i = 0; i < maxiter; ++i) {
    if (q1 == q0) return 0.5 * (p0 + p1);
    double p = p1 - q1 * (p1 - p0) / (q1 - q0);
    if (std::fabs(p - p1) < tol) return p;
    p0 = p1; q0 = q1; p1 = p; q1 = f(p1);
  }
  return p1;
}

}  // namespace scipp::optimize
