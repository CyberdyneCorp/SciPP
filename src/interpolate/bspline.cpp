// B-spline representation (BSpline) evaluated by the de Boor recursion, plus the
// default interpolating builder make_interp_spline (collocation solve) and the
// splev evaluation wrapper. Mirrors scipy.interpolate's (t, c, k) tuple.
#include "scipp/interpolate/interpolate.hpp"

#include <cmath>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::interpolate {
namespace {
namespace sd = scipp::linalg::detail;

// Knot span index l with t[l] <= x < t[l+1], clamped to [k, n-1] so that
// out-of-range x extrapolates off the boundary segment (scipy find_interval).
int find_span(const std::vector<double>& t, int k, int n, double x) {
  int l = k + 1;
  while (x >= t[l] && l != n) ++l;
  return l - 1;
}

// Non-zero B-spline basis values B_{l-k+m}(x), m=0..k, at span l (NURBS BasisFuns).
void basis_funcs(const std::vector<double>& t, int k, int l, double x,
                 std::vector<double>& out) {
  std::vector<double> left(k + 1), right(k + 1);
  out[0] = 1.0;
  for (int j = 1; j <= k; ++j) {
    left[j] = x - t[l + 1 - j];
    right[j] = t[l + j] - x;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      double temp = out[r] / (right[r + 1] + left[j - r]);
      out[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    out[j] = saved;
  }
}

// scipy's default not-a-knot knot vector for interpolation (degrees 1, 2, odd k).
std::vector<double> build_knots(const std::vector<double>& x, int k) {
  int n = static_cast<int>(x.size());
  if (n < k + 1)
    throw scipp::value_error("make_interp_spline: need at least k+1 data points");
  std::vector<double> t;
  if (k == 1) {
    t.push_back(x.front());
    t.insert(t.end(), x.begin(), x.end());
    t.push_back(x.back());
  } else if (k == 2) {
    for (int i = 0; i < 3; ++i) t.push_back(x.front());
    for (int i = 1; i < n - 2; ++i) t.push_back(0.5 * (x[i] + x[i + 1]));  // mid[1:-1]
    for (int i = 0; i < 3; ++i) t.push_back(x.back());
  } else if (k % 2 == 1) {  // not-a-knot, odd degree
    int m = (k - 1) / 2;
    for (int i = 0; i < k + 1; ++i) t.push_back(x.front());
    for (int i = m + 1; i < n - m - 1; ++i) t.push_back(x[i]);
    for (int i = 0; i < k + 1; ++i) t.push_back(x.back());
  } else {
    throw scipp::value_error("make_interp_spline: even degree > 2 not supported");
  }
  return t;
}
}  // namespace

BSpline::BSpline(const ndarray& t, const ndarray& c, int k, bool extrapolate)
    : t_(sd::to_vec(t)), c_(sd::to_vec(c)), k_(k), extrapolate_(extrapolate) {
  if (k < 0) throw scipp::value_error("BSpline: degree must be non-negative");
  int n = static_cast<int>(t_.size()) - k - 1;  // number of basis functions
  if (n < 1 || static_cast<int>(c_.size()) < n)
    throw scipp::value_error("BSpline: need len(t) >= len(c)+k+1 with len(c) >= 1");
}

double BSpline::eval_one(double x) const {
  int k = k_;
  int n = static_cast<int>(t_.size()) - k - 1;
  if (!extrapolate_ && (x < t_[k] || x > t_[n])) return std::nan("");
  int l = find_span(t_, k, n, x);
  std::vector<double> d(k + 1);
  for (int j = 0; j <= k; ++j) d[j] = c_[j + l - k];
  for (int r = 1; r <= k; ++r) {
    for (int j = k; j >= r; --j) {
      double ta = t_[j + l - k], tb = t_[j + 1 + l - r];
      double alpha = (tb > ta) ? (x - ta) / (tb - ta) : 0.0;
      d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
    }
  }
  return d[k];
}

double BSpline::operator()(double x) const { return eval_one(x); }

ndarray BSpline::operator()(const ndarray& x) const {
  std::vector<double> q = sd::to_vec(x);
  std::vector<double> out(q.size());
  for (size_t i = 0; i < q.size(); ++i) out[i] = eval_one(q[i]);
  return sd::from_vec(out);
}

ndarray BSpline::t() const { return sd::from_vec(t_); }
ndarray BSpline::c() const { return sd::from_vec(c_); }

BSpline make_interp_spline(const ndarray& x, const ndarray& y, int k) {
  std::vector<double> xv = sd::to_vec(x), yv = sd::to_vec(y);
  int n = static_cast<int>(xv.size());
  if (static_cast<int>(yv.size()) != n)
    throw scipp::value_error("make_interp_spline: x and y must have equal length");
  std::vector<double> t = build_knots(xv, k);

  // Collocation matrix A[i][j] = B_j(x_i), solved for the coefficients.
  std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
  std::vector<double> nb(k + 1);
  for (int i = 0; i < n; ++i) {
    int l = find_span(t, k, n, xv[i]);
    basis_funcs(t, k, l, xv[i], nb);
    for (int m = 0; m <= k; ++m) A[i * n + (l - k + m)] = nb[m];
  }
  ndarray cv = numpp::linalg::solve(sd::from_mat(A, n, n), sd::from_vec(yv));
  return BSpline(sd::from_vec(t), cv, k);
}

ndarray splev(const ndarray& x, const BSpline& tck) { return tck(x); }

}  // namespace scipp::interpolate
