// Summary statistics: gmean, hmean, moment, skew, kurtosis, sem, variation,
// iqr, zscore, rankdata, mode, describe.
#include "scypp/stats/stats.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

#include "scypp/linalg/detail.hpp"

namespace scypp::stats {
namespace {
namespace sd = scypp::linalg::detail;

double mean_of(const std::vector<double>& v) {
  double s = 0; for (double x : v) s += x; return s / v.size();
}
double var_of(const std::vector<double>& v, int ddof) {
  double m = mean_of(v), s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return s / (v.size() - ddof);
}
double central_moment(const std::vector<double>& v, int k) {
  double m = mean_of(v), s = 0;
  for (double x : v) s += std::pow(x - m, k);
  return s / v.size();
}
double percentile_linear(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  double rank = p / 100.0 * (v.size() - 1);
  int lo = static_cast<int>(std::floor(rank));
  int hi = static_cast<int>(std::ceil(rank));
  if (lo == hi) return v[lo];
  return v[lo] + (rank - lo) * (v[hi] - v[lo]);
}
}  // namespace

double gmean(const ndarray& a) {
  auto v = sd::to_vec(a);
  double s = 0; for (double x : v) s += std::log(x);
  return std::exp(s / v.size());
}
double hmean(const ndarray& a) {
  auto v = sd::to_vec(a);
  double s = 0; for (double x : v) s += 1.0 / x;
  return v.size() / s;
}
double moment(const ndarray& a, int order) {
  if (order == 1) return 0.0;
  return central_moment(sd::to_vec(a), order);
}
double skew(const ndarray& a, bool bias) {
  auto v = sd::to_vec(a);
  double m2 = central_moment(v, 2), m3 = central_moment(v, 3);
  double g1 = m3 / std::pow(m2, 1.5);
  if (bias) return g1;
  double n = v.size();
  return g1 * std::sqrt(n * (n - 1)) / (n - 2);
}
double kurtosis(const ndarray& a, bool fisher, bool bias) {
  auto v = sd::to_vec(a);
  double m2 = central_moment(v, 2), m4 = central_moment(v, 4);
  double g2 = m4 / (m2 * m2);
  if (!bias) {
    double n = v.size();
    g2 = 1.0 / (n - 2) / (n - 3) * ((n * n - 1.0) * (m4 / (m2 * m2)) - 3.0 * (n - 1) * (n - 1)) + 3.0;
  }
  return fisher ? g2 - 3.0 : g2;
}
double sem(const ndarray& a) {
  auto v = sd::to_vec(a);
  return std::sqrt(var_of(v, 1) / v.size());
}
double variation(const ndarray& a) {
  auto v = sd::to_vec(a);
  return std::sqrt(var_of(v, 0)) / mean_of(v);
}
double iqr(const ndarray& a) {
  auto v = sd::to_vec(a);
  return percentile_linear(v, 75.0) - percentile_linear(v, 25.0);
}
ndarray zscore(const ndarray& a) {
  auto v = sd::to_vec(a);
  double m = mean_of(v), s = std::sqrt(var_of(v, 0));
  for (double& x : v) x = (x - m) / s;
  return sd::from_vec(v);
}
ndarray rankdata(const ndarray& a) {
  auto v = sd::to_vec(a);
  int n = static_cast<int>(v.size());
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int i, int j) { return v[i] < v[j]; });
  std::vector<double> r(n);
  int i = 0;
  while (i < n) {
    int j = i;
    while (j < n && v[idx[j]] == v[idx[i]]) ++j;
    double avg = 0.5 * (i + 1 + j);  // average of ranks [i+1 .. j]
    for (int k = i; k < j; ++k) r[idx[k]] = avg;
    i = j;
  }
  return sd::from_vec(r);
}
double mode(const ndarray& a) {
  auto v = sd::to_vec(a);
  std::map<double, int> counts;
  for (double x : v) ++counts[x];
  double best = v[0]; int bestc = 0;
  for (auto& [val, c] : counts)
    if (c > bestc) { bestc = c; best = val; }
  return best;
}
DescribeResult describe(const ndarray& a) {
  auto v = sd::to_vec(a);
  DescribeResult r;
  r.nobs = static_cast<int>(v.size());
  r.minv = *std::min_element(v.begin(), v.end());
  r.maxv = *std::max_element(v.begin(), v.end());
  r.mean = mean_of(v);
  r.variance = var_of(v, 1);
  r.skewness = skew(a, true);
  r.kurtosis = kurtosis(a, true, true);
  return r;
}

}  // namespace scypp::stats
