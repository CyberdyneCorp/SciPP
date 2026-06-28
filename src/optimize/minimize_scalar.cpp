// Scalar minimization: minimize_scalar with "brent" (bracket + parabolic/golden)
// and "bounded" (fminbound) — ported from scipy.optimize.
#include "scipp/optimize/optimize.hpp"

#include <cmath>

#include "scipp/error.hpp"

namespace scipp::optimize {
namespace {

// Downhill bracketing triple (xa, xb, xc) with f(xb) < f(xa), f(xc).
void bracket(const ScalarFn& f, double& xa, double& xb, double& xc, double& fa, double& fb,
             double& fc, int& nfev) {
  const double gold = 1.618033988749895, glimit = 110.0, tiny = 1e-21;
  xa = 0.0; xb = 1.0;
  fa = f(xa); fb = f(xb);
  nfev += 2;
  if (fa < fb) { std::swap(xa, xb); std::swap(fa, fb); }
  xc = xb + gold * (xb - xa);
  fc = f(xc);
  ++nfev;
  while (fb >= fc) {
    double tmp1 = (xb - xa) * (fb - fc);
    double tmp2 = (xb - xc) * (fb - fa);
    double val = tmp2 - tmp1;
    double denom = (std::fabs(val) < tiny) ? 2.0 * tiny : 2.0 * val;
    double w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
    double wlim = xb + glimit * (xc - xb);
    double fw;
    if ((w - xc) * (xb - w) > 0.0) {
      fw = f(w); ++nfev;
      if (fw < fc) { xa = xb; xb = w; fa = fb; fb = fw; return; }
      else if (fw > fb) { xc = w; fc = fw; return; }
      w = xc + gold * (xc - xb); fw = f(w); ++nfev;
    } else if ((w - wlim) * (wlim - xc) >= 0.0) {
      w = wlim; fw = f(w); ++nfev;
    } else if ((w - wlim) * (xc - w) > 0.0) {
      fw = f(w); ++nfev;
      if (fw < fc) { xb = xc; xc = w; w = xc + gold * (xc - xb); fb = fc; fc = fw; fw = f(w); ++nfev; }
    } else {
      w = xc + gold * (xc - xb); fw = f(w); ++nfev;
    }
    xa = xb; xb = xc; xc = w; fa = fb; fb = fc; fc = fw;
  }
}

ScalarMinResult brent(const ScalarFn& f, double xtol, int maxiter) {
  const double mintol = 1e-11, cg = 0.3819660112501051;
  double xa, xb, xc, fa, fb, fc;
  int nfev = 0;
  bracket(f, xa, xb, xc, fa, fb, fc, nfev);
  double x = xb, w = xb, v = xb;
  double fx = fb, fw = fb, fv = fb;
  double a = std::min(xa, xc), b = std::max(xa, xc);
  double deltax = 0.0, rat = 0.0;
  int it = 0;
  for (; it < maxiter; ++it) {
    double tol1 = xtol * std::fabs(x) + mintol, tol2 = 2.0 * tol1;
    double xmid = 0.5 * (a + b);
    if (std::fabs(x - xmid) < (tol2 - 0.5 * (b - a))) break;
    if (std::fabs(deltax) <= tol1) {
      deltax = (x >= xmid) ? a - x : b - x;
      rat = cg * deltax;
    } else {
      double t1 = (x - w) * (fx - fv), t2 = (x - v) * (fx - fw);
      double p = (x - v) * t2 - (x - w) * t1;
      t2 = 2.0 * (t2 - t1);
      if (t2 > 0.0) p = -p;
      t2 = std::fabs(t2);
      double dx_temp = deltax;
      deltax = rat;
      if (p > t2 * (a - x) && p < t2 * (b - x) && std::fabs(p) < std::fabs(0.5 * t2 * dx_temp)) {
        rat = p / t2;
        double u = x + rat;
        if ((u - a) < tol2 || (b - u) < tol2) rat = (xmid - x >= 0.0) ? tol1 : -tol1;
      } else {
        deltax = (x >= xmid) ? a - x : b - x;
        rat = cg * deltax;
      }
    }
    double u = (std::fabs(rat) >= tol1) ? x + rat : x + ((rat >= 0.0) ? tol1 : -tol1);
    double fu = f(u);
    ++nfev;
    if (fu > fx) {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) { v = w; w = u; fv = fw; fw = fu; }
      else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
    } else {
      if (u >= x) a = x; else b = x;
      v = w; w = x; x = u; fv = fw; fw = fx; fx = fu;
    }
  }
  return {x, fx, it < maxiter, it};
}

ScalarMinResult bounded(const ScalarFn& f, double x1, double x2, double xtol, int maxiter) {
  if (x1 > x2) throw scipp::value_error("minimize_scalar: invalid bounds");
  const double sqrt_eps = 1.4901161193847656e-08, golden_mean = 0.3819660112501051;
  double a = x1, b = x2;
  double fulc = a + golden_mean * (b - a), nfc = fulc, xf = fulc;
  double rat = 0.0, e = 0.0;
  double x = xf, fx = f(x);
  int num = 1;
  double fu, ffulc = fx, fnfc = fx;
  double xm = 0.5 * (a + b);
  double tol1 = sqrt_eps * std::fabs(xf) + xtol / 3.0, tol2 = 2.0 * tol1;
  while (std::fabs(xf - xm) > (tol2 - 0.5 * (b - a))) {
    bool golden = true;
    if (std::fabs(e) > tol1) {  // parabolic fit
      golden = false;
      double r = (xf - nfc) * (fx - ffulc);
      double q = (xf - fulc) * (fx - fnfc);
      double p = (xf - fulc) * q - (xf - nfc) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = std::fabs(q);
      double e_tmp = e;
      e = rat;
      if (std::fabs(p) < std::fabs(0.5 * q * e_tmp) && p > q * (a - xf) && p < q * (b - xf)) {
        rat = p / q;
        x = xf + rat;
        if ((x - a) < tol2 || (b - x) < tol2) rat = (xm - xf >= 0.0) ? tol1 : -tol1;
      } else {
        golden = true;
      }
    }
    if (golden) {
      e = (xf >= xm) ? a - xf : b - xf;
      rat = golden_mean * e;
    }
    x = xf + ((std::fabs(rat) >= tol1) ? rat : (rat > 0.0 ? tol1 : -tol1));
    fu = f(x);
    ++num;
    if (fu <= fx) {
      if (x >= xf) a = xf; else b = xf;
      fulc = nfc; ffulc = fnfc; nfc = xf; fnfc = fx; xf = x; fx = fu;
    } else {
      if (x < xf) a = x; else b = x;
      if (fu <= fnfc || nfc == xf) { fulc = nfc; ffulc = fnfc; nfc = x; fnfc = fu; }
      else if (fu <= ffulc || fulc == xf || fulc == nfc) { fulc = x; ffulc = fu; }
    }
    xm = 0.5 * (a + b);
    tol1 = sqrt_eps * std::fabs(xf) + xtol / 3.0;
    tol2 = 2.0 * tol1;
    if (num >= maxiter) break;
  }
  return {xf, fx, num < maxiter, num};
}

}  // namespace

ScalarMinResult minimize_scalar(const ScalarFn& f, const std::string& method,
                                std::optional<std::pair<double, double>> bounds, double xtol,
                                int maxiter) {
  if (method == "brent") return brent(f, xtol, maxiter);
  if (method == "bounded") {
    if (!bounds) throw scipp::value_error("minimize_scalar(bounded) requires bounds");
    return bounded(f, bounds->first, bounds->second, xtol, maxiter);
  }
  throw scipp::value_error("minimize_scalar: unknown method " + method);
}

}  // namespace scipp::optimize
