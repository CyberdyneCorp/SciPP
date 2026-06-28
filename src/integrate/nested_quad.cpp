// Nested and extended quadrature built on the adaptive `quad`: romberg,
// quad_vec, dblquad, tplquad, nquad — mirroring scipy.integrate.
#include "scipp/integrate/integrate.hpp"

#include <cmath>
#include <functional>
#include <vector>

#include "numpp/core/dtype.hpp"

namespace scipp::integrate {

double romberg(const Integrand& f, double a, double b, double tol, double rtol, int divmax) {
  std::vector<double> prev(divmax + 1), cur(divmax + 1);
  double h = b - a;
  prev[0] = 0.5 * h * (f(a) + f(b));
  double last = prev[0];
  for (int i = 1; i <= divmax; ++i) {
    h *= 0.5;
    // Composite-trapezoid refinement: add the new midpoints only.
    double s = 0.0;
    int npts = 1 << (i - 1);
    for (int k = 1; k <= npts; ++k) s += f(a + (2 * k - 1) * h);
    cur[0] = 0.5 * prev[0] + h * s;
    double factor = 4.0;
    for (int j = 1; j <= i; ++j) {
      cur[j] = cur[j - 1] + (cur[j - 1] - prev[j - 1]) / (factor - 1.0);
      factor *= 4.0;
    }
    if (i > 1 && std::fabs(cur[i] - last) <= std::max(tol, rtol * std::fabs(cur[i])))
      return cur[i];
    last = cur[i];
    std::swap(prev, cur);
  }
  return last;
}

ndarray quad_vec(const VecIntegrand& f, double a, double b, double epsabs, double epsrel,
                 int limit) {
  int k = static_cast<int>(f(a).size());
  numpp::ndarray out(numpp::Shape{k}, numpp::kFloat64);
  double* o = out.typed_data<double>();
  for (int c = 0; c < k; ++c) {
    Integrand comp = [&, c](double x) { return f(x).typed_data<double>()[c]; };
    o[c] = quad(comp, a, b, epsabs, epsrel, limit).value;
  }
  return out;
}

QuadResult dblquad(const Integrand2& f, double a, double b, const Bound1& gfun,
                   const Bound1& hfun, double epsabs, double epsrel) {
  Integrand outer = [&](double x) {
    Integrand inner = [&, x](double y) { return f(y, x); };
    return quad(inner, gfun(x), hfun(x), epsabs, epsrel).value;
  };
  return quad(outer, a, b, epsabs, epsrel);
}

QuadResult tplquad(const Integrand3& f, double a, double b, const Bound1& gfun,
                   const Bound1& hfun, const Bound2& qfun, const Bound2& rfun,
                   double epsabs, double epsrel) {
  Integrand outer = [&](double x) {
    Integrand mid = [&, x](double y) {
      Integrand inner = [&, x, y](double z) { return f(z, y, x); };
      return quad(inner, qfun(x, y), rfun(x, y), epsabs, epsrel).value;
    };
    return quad(mid, gfun(x), hfun(x), epsabs, epsrel).value;
  };
  return quad(outer, a, b, epsabs, epsrel);
}

QuadResult nquad(const IntegrandN& f, const std::vector<std::pair<double, double>>& ranges,
                 double epsabs, double epsrel) {
  int n = static_cast<int>(ranges.size());
  std::vector<double> pt(n);
  // Integrate dimension `level` and recurse on the inner dimensions.
  std::function<double(int)> rec = [&](int level) -> double {
    if (level == n) return f(pt);
    Integrand g = [&, level](double x) {
      pt[level] = x;
      return rec(level + 1);
    };
    return quad(g, ranges[level].first, ranges[level].second, epsabs, epsrel).value;
  };
  Integrand top = [&](double x) {
    pt[0] = x;
    return rec(1);
  };
  return quad(top, ranges[0].first, ranges[0].second, epsabs, epsrel);
}

}  // namespace scipp::integrate
