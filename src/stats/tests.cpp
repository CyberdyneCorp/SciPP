// Correlation/regression and parametric hypothesis tests.
#include "scipp/stats/stats.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::stats {
namespace {
namespace sd = scipp::linalg::detail;

double mean_of(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s / v.size(); }
double ss_centered(const std::vector<double>& v, double m) {
  double s = 0; for (double x : v) s += (x - m) * (x - m); return s;
}
int sgn(double v) { return (v > 0) - (v < 0); }

// Exact two-sided Kolmogorov CDF P(D_n < d) — Marsaglia, Tsang & Wang (2003),
// matching scipy.stats.kstwo. Returns the CDF; the survival is 1 - K.
void mmul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) {
      double s = 0; for (int k = 0; k < m; ++k) s += A[i * m + k] * B[k * m + j];
      C[i * m + j] = s;
    }
}
void mpow(const std::vector<double>& A, int eA, std::vector<double>& V, int& eV, int m, int n) {
  if (n == 1) { V = A; eV = eA; return; }
  mpow(A, eA, V, eV, m, n / 2);
  std::vector<double> B(m * m);
  mmul(V, V, B, m);
  int eB = 2 * eV;
  if (n % 2 == 0) { V = B; eV = eB; }
  else { std::vector<double> C(m * m); mmul(A, B, C, m); V = C; eV = eA + eB; }
  if (V[(m / 2) * m + (m / 2)] > 1e140) { for (double& x : V) x *= 1e-140; eV += 140; }
}
double ks_cdf(int n, double d) {
  int k = static_cast<int>(n * d) + 1, m = 2 * k - 1;
  double h = k - n * d;
  std::vector<double> H(m * m), Q;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) H[i * m + j] = (i - j + 1 < 0) ? 0.0 : 1.0;
  for (int i = 0; i < m; ++i) {
    H[i * m + 0] -= std::pow(h, i + 1);
    H[(m - 1) * m + i] -= std::pow(h, m - i);
  }
  H[(m - 1) * m + 0] += (2.0 * h - 1.0 > 0) ? std::pow(2.0 * h - 1.0, m) : 0.0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      if (i - j + 1 > 0)
        for (int g = 1; g <= i - j + 1; ++g) H[i * m + j] /= g;
  int eH = 0, eQ;
  mpow(H, eH, Q, eQ, m, n);
  double s = Q[(k - 1) * m + (k - 1)];
  for (int i = 1; i <= n; ++i) { s = s * i / n; if (s < 1e-140) { s *= 1e140; eQ -= 140; } }
  return s * std::pow(10.0, static_cast<double>(eQ));
}

double pearson_r(const std::vector<double>& x, const std::vector<double>& y) {
  double mx = mean_of(x), my = mean_of(y), sxy = 0;
  for (size_t i = 0; i < x.size(); ++i) sxy += (x[i] - mx) * (y[i] - my);
  return sxy / std::sqrt(ss_centered(x, mx) * ss_centered(y, my));
}

double skewtest_z(const ndarray& a) {
  auto v = sd::to_vec(a);
  double n = v.size();
  double b2 = skew(a, true);
  double Y = b2 * std::sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)));
  double beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) /
                 ((n - 2) * (n + 5) * (n + 7) * (n + 9));
  double W2 = -1.0 + std::sqrt(2.0 * (beta2 - 1.0));
  double delta = 1.0 / std::sqrt(0.5 * std::log(W2));
  double alpha = std::sqrt(2.0 / (W2 - 1.0));
  return delta * std::asinh(Y / alpha);
}
double kurtosistest_z(const ndarray& a) {
  auto v = sd::to_vec(a);
  double n = v.size();
  double b2 = kurtosis(a, false, true);  // m4/m2^2
  double E = 3.0 * (n - 1) / (n + 1);
  double varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
  double x = (b2 - E) / std::sqrt(varb2);
  double sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) *
                     std::sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)));
  double A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + std::sqrt(1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1)));
  double term1 = 1.0 - 2.0 / (9.0 * A);
  double denom = 1.0 + x * std::sqrt(2.0 / (A - 4.0));
  double term2 = sgn(denom) * std::cbrt((1.0 - 2.0 / A) / std::fabs(denom));
  return (term1 - term2) / std::sqrt(2.0 / (9.0 * A));
}
}  // namespace

CorrResult pearsonr(const ndarray& x, const ndarray& y) {
  auto xv = sd::to_vec(x), yv = sd::to_vec(y);
  double r = std::clamp(pearson_r(xv, yv), -1.0, 1.0);
  double n = xv.size(), ab = n / 2.0 - 1.0;
  double p = 2.0 * beta::sf(std::fabs(r), ab, ab, -1.0, 2.0);
  return {r, std::min(1.0, p)};
}

CorrResult spearmanr(const ndarray& x, const ndarray& y) {
  auto rx = sd::to_vec(rankdata(x)), ry = sd::to_vec(rankdata(y));
  double rs = pearson_r(rx, ry);
  double n = rx.size(), df = n - 2;
  double tstat = rs * std::sqrt(df / ((1.0 - rs) * (1.0 + rs)));
  double p = 2.0 * t::sf(std::fabs(tstat), df);
  return {rs, p};
}

LinregressResult linregress(const ndarray& x, const ndarray& y) {
  auto xv = sd::to_vec(x), yv = sd::to_vec(y);
  double n = xv.size(), mx = mean_of(xv), my = mean_of(yv);
  double ssxm = 0, ssym = 0, ssxym = 0;
  for (size_t i = 0; i < xv.size(); ++i) {
    ssxm += (xv[i] - mx) * (xv[i] - mx);
    ssym += (yv[i] - my) * (yv[i] - my);
    ssxym += (xv[i] - mx) * (yv[i] - my);
  }
  ssxm /= n; ssym /= n; ssxym /= n;
  double slope = ssxym / ssxm;
  double intercept = my - slope * mx;
  double r = std::clamp(ssxym / std::sqrt(ssxm * ssym), -1.0, 1.0);
  double df = n - 2;
  double tstat = r * std::sqrt(df / ((1.0 - r) * (1.0 + r)));
  double p = 2.0 * t::sf(std::fabs(tstat), df);
  double slope_se = std::sqrt((1.0 - r * r) * ssym / ssxm / df);
  double intercept_se = slope_se * std::sqrt(ssxm + mx * mx);
  return {slope, intercept, r, p, slope_se, intercept_se};
}

TestResult ttest_1samp(const ndarray& a, double popmean) {
  auto v = sd::to_vec(a);
  double n = v.size(), m = mean_of(v);
  double s2 = ss_centered(v, m) / (n - 1);
  double tstat = (m - popmean) / std::sqrt(s2 / n);
  return {tstat, 2.0 * t::sf(std::fabs(tstat), n - 1)};
}
TestResult ttest_ind(const ndarray& a, const ndarray& b) {
  auto av = sd::to_vec(a), bv = sd::to_vec(b);
  double na = av.size(), nb = bv.size(), ma = mean_of(av), mb = mean_of(bv);
  double sp2 = (ss_centered(av, ma) + ss_centered(bv, mb)) / (na + nb - 2);
  double tstat = (ma - mb) / std::sqrt(sp2 * (1.0 / na + 1.0 / nb));
  double df = na + nb - 2;
  return {tstat, 2.0 * t::sf(std::fabs(tstat), df)};
}
TestResult ttest_rel(const ndarray& a, const ndarray& b) {
  auto av = sd::to_vec(a), bv = sd::to_vec(b);
  std::vector<double> d(av.size());
  for (size_t i = 0; i < av.size(); ++i) d[i] = av[i] - bv[i];
  return ttest_1samp(sd::from_vec(d), 0.0);
}

TestResult f_oneway(const std::vector<ndarray>& groups) {
  std::vector<std::vector<double>> g;
  double total = 0, N = 0;
  for (auto& a : groups) { g.push_back(sd::to_vec(a)); for (double x : g.back()) { total += x; ++N; } }
  double grand = total / N;
  double ssb = 0, ssw = 0;
  for (auto& gi : g) {
    double mi = mean_of(gi);
    ssb += gi.size() * (mi - grand) * (mi - grand);
    ssw += ss_centered(gi, mi);
  }
  double dfb = g.size() - 1, dfw = N - g.size();
  double F = (ssb / dfb) / (ssw / dfw);
  return {F, f::sf(F, dfb, dfw)};
}

TestResult ks_2samp(const ndarray& a, const ndarray& b) {
  auto av = sd::to_vec(a), bv = sd::to_vec(b);
  std::sort(av.begin(), av.end());
  std::sort(bv.begin(), bv.end());
  std::vector<double> all = av;
  all.insert(all.end(), bv.begin(), bv.end());
  std::sort(all.begin(), all.end());
  double na = av.size(), nb = bv.size(), D = 0;
  for (double x : all) {
    double f1 = (std::upper_bound(av.begin(), av.end(), x) - av.begin()) / na;
    double f2 = (std::upper_bound(bv.begin(), bv.end(), x) - bv.begin()) / nb;
    D = std::max(D, std::fabs(f1 - f2));
  }
  double en = na * nb / (na + nb);
  int n = static_cast<int>(std::nearbyint(en));  // round-half-to-even, as scipy
  double p = 1.0 - ks_cdf(n, D);  // scipy method="asymp": kstwo.sf(D, round(en))
  return {D, std::clamp(p, 0.0, 1.0)};
}

Chi2ContingencyResult chi2_contingency(const ndarray& table) {
  numpp::ndarray T = table.astype(numpp::kFloat64).ascontiguousarray();
  int r = static_cast<int>(T.shape()[0]), c = static_cast<int>(T.shape()[1]);
  const double* o = T.typed_data<double>();
  std::vector<double> rs(r, 0), cs(c, 0); double tot = 0;
  for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) { rs[i] += o[i * c + j]; cs[j] += o[i * c + j]; tot += o[i * c + j]; }
  std::vector<double> exp(r * c);
  bool yates = (r == 2 && c == 2);
  double stat = 0;
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) {
      double e = rs[i] * cs[j] / tot;
      exp[i * c + j] = e;
      double diff = std::fabs(o[i * c + j] - e);
      if (yates) diff = std::max(0.0, diff - 0.5);
      stat += diff * diff / e;
    }
  int dof = (r - 1) * (c - 1);
  return {stat, chi2::sf(stat, dof), dof, sd::from_mat(exp, r, c)};
}

TestResult normaltest(const ndarray& a) {
  double zs = skewtest_z(a), zk = kurtosistest_z(a);
  double stat = zs * zs + zk * zk;
  return {stat, chi2::sf(stat, 2)};
}

}  // namespace scipp::stats
