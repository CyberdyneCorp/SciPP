// Nonparametric rank tests: mannwhitneyu, wilcoxon, kruskal, kendalltau
// (asymptotic / normal-approximation p-values with tie correction).
#include "scipp/stats/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "scipp/linalg/detail.hpp"

namespace scipp::stats {
namespace {
namespace sd = scipp::linalg::detail;
int sgn(double v) { return (v > 0) - (v < 0); }

// average ranks; also returns sum over tie groups of (t^3 - t).
std::vector<double> avg_ranks(const std::vector<double>& v, double& tie_sum) {
  int n = static_cast<int>(v.size());
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] < v[b]; });
  std::vector<double> r(n);
  tie_sum = 0;
  int i = 0;
  while (i < n) {
    int j = i;
    while (j < n && v[idx[j]] == v[idx[i]]) ++j;
    double avg = 0.5 * (i + 1 + j);
    double t = j - i;
    tie_sum += t * t * t - t;
    for (int k = i; k < j; ++k) r[idx[k]] = avg;
    i = j;
  }
  return r;
}
}  // namespace

TestResult mannwhitneyu(const ndarray& x, const ndarray& y) {
  auto xv = sd::to_vec(x), yv = sd::to_vec(y);
  double n1 = xv.size(), n2 = yv.size(), N = n1 + n2;
  std::vector<double> all = xv;
  all.insert(all.end(), yv.begin(), yv.end());
  double ties;
  auto r = avg_ranks(all, ties);
  double R1 = 0; for (int i = 0; i < n1; ++i) R1 += r[i];
  double U1 = R1 - n1 * (n1 + 1) / 2.0;
  double mu = n1 * n2 / 2.0;
  double sigma = std::sqrt(n1 * n2 / 12.0 * ((N + 1) - ties / (N * (N - 1))));
  double num = U1 - mu;
  num -= sgn(num) * 0.5;  // continuity correction
  double z = num / sigma;
  return {U1, std::min(1.0, 2.0 * norm::sf(std::fabs(z)))};
}

TestResult wilcoxon(const ndarray& x, const ndarray& y) {
  auto xv = sd::to_vec(x), yv = sd::to_vec(y);
  std::vector<double> diff, absd;
  for (size_t i = 0; i < xv.size(); ++i) { double d = xv[i] - yv[i]; if (d != 0) { diff.push_back(d); absd.push_back(std::fabs(d)); } }
  double n = diff.size(), ties;
  auto r = avg_ranks(absd, ties);
  double rp = 0, rm = 0;
  for (size_t i = 0; i < diff.size(); ++i) (diff[i] > 0 ? rp : rm) += r[i];
  double T = std::min(rp, rm);
  double mn = n * (n + 1) / 4.0;
  double se = std::sqrt((n * (n + 1) * (2 * n + 1) - ties / 2.0) / 24.0);
  double z = (T - mn) / se;  // scipy wilcoxon default: correction=False
  return {T, std::min(1.0, 2.0 * norm::sf(std::fabs(z)))};
}

TestResult kruskal(const std::vector<ndarray>& groups) {
  std::vector<double> all;
  std::vector<int> sizes;
  for (auto& g : groups) { auto v = sd::to_vec(g); sizes.push_back((int)v.size()); all.insert(all.end(), v.begin(), v.end()); }
  double N = all.size(), ties;
  auto r = avg_ranks(all, ties);
  double H = 0; int off = 0;
  for (int s : sizes) { double Rsum = 0; for (int i = 0; i < s; ++i) Rsum += r[off + i]; H += Rsum * Rsum / s; off += s; }
  H = 12.0 / (N * (N + 1)) * H - 3.0 * (N + 1);
  H /= (1.0 - ties / (N * N * N - N));  // tie correction
  return {H, chi2::sf(H, groups.size() - 1.0)};
}

CorrResult kendalltau(const ndarray& x, const ndarray& y) {
  auto xv = sd::to_vec(x), yv = sd::to_vec(y);
  int n = static_cast<int>(xv.size());
  double P = 0, Q = 0, Tx = 0, Ty = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      int sx = sgn(xv[i] - xv[j]), sy = sgn(yv[i] - yv[j]);
      if (sx == 0 && sy == 0) continue;
      if (sx == 0) { ++Tx; continue; }
      if (sy == 0) { ++Ty; continue; }
      if (sx * sy > 0) ++P; else ++Q;
    }
  double tau = (P - Q) / std::sqrt((P + Q + Tx) * (P + Q + Ty));
  // asymptotic normal approximation (no-tie variance)
  double var = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0));
  double z = tau / std::sqrt(var);
  return {tau, std::min(1.0, 2.0 * norm::sf(std::fabs(z)))};
}

}  // namespace scipp::stats
