// Quadrature: trapezoid, simpson, cumulative_trapezoid, fixed_quad
// (Gauss-Legendre) and quad (adaptive Gauss-Kronrod 21-point).
#include "scypp/integrate/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::integrate {
namespace {
namespace sd = scypp::linalg::detail;
constexpr double kPi = 3.141592653589793238462643383279502884;
}  // namespace

double trapezoid(const ndarray& y, std::optional<ndarray> x, double dx) {
  std::vector<double> v = sd::to_vec(y);
  int n = static_cast<int>(v.size());
  double s = 0.0;
  if (x) {
    std::vector<double> xv = sd::to_vec(*x);
    for (int i = 0; i < n - 1; ++i) s += 0.5 * (v[i] + v[i + 1]) * (xv[i + 1] - xv[i]);
  } else {
    for (int i = 0; i < n - 1; ++i) s += 0.5 * (v[i] + v[i + 1]) * dx;
  }
  return s;
}

double simpson(const ndarray& y, std::optional<ndarray> x, double dx) {
  std::vector<double> v = sd::to_vec(y);
  int N = static_cast<int>(v.size());
  if (N < 2) return 0.0;
  std::vector<double> xv;
  if (x) xv = sd::to_vec(*x);
  auto h = [&](int i) { return x ? (xv[i + 1] - xv[i]) : dx; };

  // Composite Simpson over an even number of intervals [start, start+2].
  auto pair_simpson = [&](int i) {
    double h0 = h(i), h1 = h(i + 1);
    double hph = h0 + h1, hdh = h1 / h0, hmh = h0 * h1;
    return hph / 6.0 * ((2.0 - hdh) * v[i] + (hph * hph / hmh) * v[i + 1] + (2.0 - 1.0 / hdh) * v[i + 2]);
  };

  double result = 0.0;
  int nintervals = N - 1;
  int stop = (nintervals % 2 == 0) ? N - 1 : N - 2;  // last index handled by the pure rule
  for (int i = 0; i + 2 < stop + 1; i += 2) result += pair_simpson(i);

  if (nintervals % 2 == 1) {  // odd #intervals: parabolic correction on the last interval
    int n = N - 1;
    double h0 = h(n - 2), h1 = h(n - 1);
    double denom = h0 + h1;
    result += v[n] * (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * denom);
    result += v[n - 1] * (h1 * h1 + 3.0 * h1 * h0) / (6.0 * h0);
    result -= v[n - 2] * (h1 * h1 * h1) / (6.0 * h0 * denom);
  }
  return result;
}

ndarray cumulative_trapezoid(const ndarray& y, std::optional<ndarray> x, double dx,
                             std::optional<double> initial) {
  std::vector<double> v = sd::to_vec(y);
  int n = static_cast<int>(v.size());
  std::vector<double> xv;
  if (x) xv = sd::to_vec(*x);
  std::vector<double> cum(n - 1);
  double acc = 0.0;
  for (int i = 0; i < n - 1; ++i) {
    double step = x ? (xv[i + 1] - xv[i]) : dx;
    acc += 0.5 * (v[i] + v[i + 1]) * step;
    cum[i] = acc;
  }
  if (initial) {
    std::vector<double> out(n);
    out[0] = *initial;
    for (int i = 0; i < n - 1; ++i) out[i + 1] = *initial + cum[i];
    return sd::from_vec(out);
  }
  return sd::from_vec(cum);
}

double fixed_quad(const Integrand& f, double a, double b, int n) {
  // Gauss-Legendre nodes/weights via Newton iteration on the Legendre polynomial.
  std::vector<double> xnode(n), w(n);
  for (int i = 0; i < n; ++i) {
    double xi = std::cos(kPi * (i + 0.75) / (n + 0.5));  // initial guess
    double dp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int k = 0; k < n; ++k) {  // Legendre recurrence
        double p2 = p1; p1 = p0;
        p0 = ((2.0 * k + 1.0) * xi * p1 - k * p2) / (k + 1.0);
      }
      dp = n * (xi * p0 - p1) / (xi * xi - 1.0);
      double dx = p0 / dp;
      xi -= dx;
      if (std::fabs(dx) < 1e-15) break;
    }
    xnode[i] = xi;
    w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
  }
  double c1 = 0.5 * (b - a), c2 = 0.5 * (b + a);
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += w[i] * f(c1 * xnode[i] + c2);
  return c1 * s;
}

namespace {
// 21-point Gauss-Kronrod rule (QUADPACK qk21 constants).
const double kXgk[11] = {
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
    0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
    0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
    0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0};
const double kWgk[11] = {
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
    0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
    0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
    0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
    0.149445554002916905664936468389821};
const double kWg[5] = {
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
    0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
    0.295524224714752870173892994651338};

struct Segment { double a, b, value, error; };

QuadResult qk21(const Integrand& f, double a, double b) {
  double centr = 0.5 * (a + b), hlgth = 0.5 * (b - a), dhlgth = std::fabs(hlgth);
  double fc = f(centr);
  double resk = kWgk[10] * fc, resg = 0.0;
  for (int j = 0; j < 5; ++j) {  // 10-point Gauss abscissae (odd kgk indices)
    int jtw = 2 * j + 1;
    double absc = hlgth * kXgk[jtw];
    double fsum = f(centr - absc) + f(centr + absc);
    resg += kWg[j] * fsum;
    resk += kWgk[jtw] * fsum;
  }
  for (int j = 0; j < 5; ++j) {  // Kronrod-only abscissae (even kgk indices)
    int jtwm1 = 2 * j;
    double absc = hlgth * kXgk[jtwm1];
    resk += kWgk[jtwm1] * (f(centr - absc) + f(centr + absc));
  }
  double value = resk * hlgth;
  double err = std::fabs((resk - resg) * hlgth);
  (void)dhlgth;
  return {value, err};
}
}  // namespace

QuadResult quad(const Integrand& f, double a, double b, double epsabs, double epsrel, int limit) {
  QuadResult q0 = qk21(f, a, b);
  std::vector<Segment> segs{{a, b, q0.value, q0.abserr}};
  double total = q0.value, toterr = q0.abserr;
  for (int iter = 0; iter < limit; ++iter) {
    if (toterr <= std::max(epsabs, epsrel * std::fabs(total))) break;
    // Bisect the worst-error segment.
    int worst = 0;
    for (int i = 1; i < static_cast<int>(segs.size()); ++i)
      if (segs[i].error > segs[worst].error) worst = i;
    Segment s = segs[worst];
    double mid = 0.5 * (s.a + s.b);
    QuadResult left = qk21(f, s.a, mid), right = qk21(f, mid, s.b);
    total += (left.value + right.value) - s.value;
    toterr += (left.abserr + right.abserr) - s.error;
    segs[worst] = {s.a, mid, left.value, left.abserr};
    segs.push_back({mid, s.b, right.value, right.abserr});
  }
  return {total, toterr};
}

}  // namespace scypp::integrate
