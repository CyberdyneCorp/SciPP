// Piecewise-cubic interpolators: CubicSpline (tridiagonal moment solve),
// PchipInterpolator (Fritsch-Carlson), Akima1DInterpolator. Shared evaluation.
#include "scypp/interpolate/interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::interpolate {
namespace {
namespace sd = scypp::linalg::detail;

int sgn(double v) { return (v > 0) - (v < 0); }

// Cubic Hermite coefficients (a + b t + c t^2 + d t^3, t = x - x_i) from node
// values y_i,y_{i+1}, slopes m_i,m_{i+1} over an interval of width h.
void hermite(double yi, double yj, double mi, double mj, double h, double* out) {
  double delta = (yj - yi) / h;
  out[0] = yi;
  out[1] = mi;
  out[2] = (3.0 * delta - 2.0 * mi - mj) / h;
  out[3] = (mi + mj - 2.0 * delta) / (h * h);
}
}  // namespace

double PiecewiseCubic::eval_scalar(double xq, int nu) const {
  int n = static_cast<int>(x_.size());
  int i = static_cast<int>(std::upper_bound(x_.begin(), x_.end(), xq) - x_.begin()) - 1;
  i = std::clamp(i, 0, n - 2);
  double t = xq - x_[i];
  const double* c = &c_[4 * i];
  switch (nu) {
    case 0: return c[0] + t * (c[1] + t * (c[2] + t * c[3]));
    case 1: return c[1] + t * (2.0 * c[2] + 3.0 * t * c[3]);
    case 2: return 2.0 * c[2] + 6.0 * t * c[3];
    case 3: return 6.0 * c[3];
    default: return 0.0;
  }
}

double PiecewiseCubic::operator()(double xq, int nu) const { return eval_scalar(xq, nu); }

ndarray PiecewiseCubic::operator()(const ndarray& xq, int nu) const {
  std::vector<double> q = sd::to_vec(xq);
  std::vector<double> out(q.size());
  for (size_t i = 0; i < q.size(); ++i) out[i] = eval_scalar(q[i], nu);
  return sd::from_vec(out);
}

CubicSpline::CubicSpline(const ndarray& x, const ndarray& y, std::string bc_type) {
  x_ = sd::to_vec(x);
  std::vector<double> yv = sd::to_vec(y);
  int n = static_cast<int>(x_.size());
  if (n < 2 || static_cast<int>(yv.size()) != n)
    throw scypp::value_error("CubicSpline: need matching x,y of length >= 2");
  std::vector<double> h(n - 1);
  for (int i = 0; i < n - 1; ++i) h[i] = x_[i + 1] - x_[i];

  std::vector<double> M(n, 0.0);
  if (n == 2) {
    M[0] = M[1] = 0.0;
  } else {
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0), rhs(n, 0.0);
    auto a = [&](int r, int c) -> double& { return A[r * n + c]; };
    for (int i = 1; i < n - 1; ++i) {
      a(i, i - 1) = h[i - 1];
      a(i, i) = 2.0 * (h[i - 1] + h[i]);
      a(i, i + 1) = h[i];
      rhs[i] = 6.0 * ((yv[i + 1] - yv[i]) / h[i] - (yv[i] - yv[i - 1]) / h[i - 1]);
    }
    if (bc_type == "natural") {
      a(0, 0) = 1.0; a(n - 1, n - 1) = 1.0;
    } else if (bc_type == "clamped") {  // zero first derivative at ends
      a(0, 0) = 2.0 * h[0]; a(0, 1) = h[0];
      rhs[0] = 6.0 * ((yv[1] - yv[0]) / h[0]);
      a(n - 1, n - 2) = h[n - 2]; a(n - 1, n - 1) = 2.0 * h[n - 2];
      rhs[n - 1] = 6.0 * (-(yv[n - 1] - yv[n - 2]) / h[n - 2]);
    } else {  // not-a-knot (default): third-derivative continuity at x_1, x_{n-2}
      a(0, 0) = -h[1]; a(0, 1) = h[0] + h[1]; a(0, 2) = -h[0];
      a(n - 1, n - 3) = -h[n - 2]; a(n - 1, n - 2) = h[n - 2] + h[n - 3]; a(n - 1, n - 1) = -h[n - 3];
    }
    M = sd::to_vec(numpp::linalg::solve(sd::from_mat(A, n, n), sd::from_vec(rhs)));
  }

  c_.assign(4 * (n - 1), 0.0);
  for (int i = 0; i < n - 1; ++i) {
    double hi = h[i];
    c_[4 * i + 0] = yv[i];
    c_[4 * i + 1] = (yv[i + 1] - yv[i]) / hi - hi * (2.0 * M[i] + M[i + 1]) / 6.0;
    c_[4 * i + 2] = M[i] / 2.0;
    c_[4 * i + 3] = (M[i + 1] - M[i]) / (6.0 * hi);
  }
}

PchipInterpolator::PchipInterpolator(const ndarray& x, const ndarray& y) {
  x_ = sd::to_vec(x);
  std::vector<double> yv = sd::to_vec(y);
  int n = static_cast<int>(x_.size());
  std::vector<double> h(n - 1), d(n - 1);
  for (int i = 0; i < n - 1; ++i) { h[i] = x_[i + 1] - x_[i]; d[i] = (yv[i + 1] - yv[i]) / h[i]; }

  std::vector<double> m(n, 0.0);
  auto edge = [](double h0, double h1, double d0, double d1) {
    double v = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (sgn(v) != sgn(d0)) v = 0.0;
    else if (sgn(d0) != sgn(d1) && std::fabs(v) > 3.0 * std::fabs(d0)) v = 3.0 * d0;
    return v;
  };
  if (n == 2) {
    m[0] = m[1] = d[0];
  } else {
    for (int i = 1; i < n - 1; ++i) {
      if (d[i - 1] * d[i] <= 0.0) m[i] = 0.0;
      else {
        double w1 = 2.0 * h[i] + h[i - 1], w2 = h[i] + 2.0 * h[i - 1];
        m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
      }
    }
    m[0] = edge(h[0], h[1], d[0], d[1]);
    m[n - 1] = edge(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
  }
  c_.assign(4 * (n - 1), 0.0);
  for (int i = 0; i < n - 1; ++i) hermite(yv[i], yv[i + 1], m[i], m[i + 1], h[i], &c_[4 * i]);
}

Akima1DInterpolator::Akima1DInterpolator(const ndarray& x, const ndarray& y) {
  x_ = sd::to_vec(x);
  std::vector<double> yv = sd::to_vec(y);
  int n = static_cast<int>(x_.size());
  std::vector<double> h(n - 1), s(n - 1);
  for (int i = 0; i < n - 1; ++i) { h[i] = x_[i + 1] - x_[i]; s[i] = (yv[i + 1] - yv[i]) / h[i]; }

  std::vector<double> mm(n + 3);  // extended secants
  for (int i = 0; i < n - 1; ++i) mm[i + 2] = s[i];
  mm[1] = 2.0 * mm[2] - mm[3];
  mm[0] = 2.0 * mm[1] - mm[2];
  mm[n + 1] = 2.0 * mm[n] - mm[n - 1];
  mm[n + 2] = 2.0 * mm[n + 1] - mm[n];

  std::vector<double> dm(n + 2);
  for (int i = 0; i < n + 2; ++i) dm[i] = std::fabs(mm[i + 1] - mm[i]);
  double fmax = 0.0;
  std::vector<double> f12(n);
  for (int i = 0; i < n; ++i) { f12[i] = dm[i + 2] + dm[i]; fmax = std::max(fmax, f12[i]); }

  std::vector<double> m(n);
  for (int i = 0; i < n; ++i) {
    double f1 = dm[i + 2], f2 = dm[i];
    if (f12[i] > 1e-9 * fmax) m[i] = (f1 * mm[i + 1] + f2 * mm[i + 2]) / f12[i];
    else m[i] = 0.5 * (mm[i + 1] + mm[i + 2]);
  }
  c_.assign(4 * (n - 1), 0.0);
  for (int i = 0; i < n - 1; ++i) hermite(yv[i], yv[i + 1], m[i], m[i + 1], h[i], &c_[4 * i]);
}

}  // namespace scypp::interpolate
