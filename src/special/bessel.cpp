// Bessel functions via the C++17 <cmath> special-math family (libstdc++),
// which matches SciPy/Cephes within tolerance. Negative arguments are declined
// to nan (SciPy returns nan for real-order jv/yv at x < 0).

#include "scipp/special/special.hpp"

#include <cmath>

#include "scipp/detail/elementwise.hpp"

namespace scipp::special {
namespace {
inline bool bad(double x) { return x < 0.0; }
}  // namespace

double jv(double v, double x) { return bad(x) ? std::nan("") : std::cyl_bessel_j(v, x); }
double yv(double v, double x) { return x <= 0.0 ? (x == 0.0 ? -INFINITY : std::nan("")) : std::cyl_neumann(v, x); }
double iv(double v, double x) { return bad(x) ? std::nan("") : std::cyl_bessel_i(v, x); }
double kv(double v, double x) { return x <= 0.0 ? (x == 0.0 ? INFINITY : std::nan("")) : std::cyl_bessel_k(v, x); }

double jn(int n, double x) { return jv(static_cast<double>(n), x); }
double yn(int n, double x) { return yv(static_cast<double>(n), x); }

double i0(double x) { return iv(0.0, x); }
double i1(double x) { return iv(1.0, x); }
double k0(double x) { return kv(0.0, x); }
double k1(double x) { return kv(1.0, x); }

ndarray jv(double v, const ndarray& x) { return detail::map(x, [v](double t) { return jv(v, t); }); }
ndarray yv(double v, const ndarray& x) { return detail::map(x, [v](double t) { return yv(v, t); }); }
ndarray iv(double v, const ndarray& x) { return detail::map(x, [v](double t) { return iv(v, t); }); }
ndarray kv(double v, const ndarray& x) { return detail::map(x, [v](double t) { return kv(v, t); }); }
ndarray i0(const ndarray& x) { return detail::map(x, [](double t) { return i0(t); }); }
ndarray i1(const ndarray& x) { return detail::map(x, [](double t) { return i1(t); }); }
ndarray k0(const ndarray& x) { return detail::map(x, [](double t) { return k0(t); }); }
ndarray k1(const ndarray& x) { return detail::map(x, [](double t) { return k1(t); }); }

}  // namespace scipp::special
