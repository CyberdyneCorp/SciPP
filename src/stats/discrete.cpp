// Discrete distributions: binom, poisson, geom, bernoulli, nbinom, hypergeom.
#include "scypp/stats/stats.hpp"

#include <algorithm>
#include <cmath>

#include "scypp/stats/detail.hpp"

namespace scypp::stats {
namespace d = detail;
namespace {
double lchoose(double n, double k) { return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1); }
}  // namespace

// ---- binom ----
double binom::logpmf(int k, int n, double p) {
  if (k < 0 || k > n) return -INFINITY;
  return lchoose(n, k) + k * std::log(p) + (n - k) * std::log1p(-p);
}
double binom::pmf(int k, int n, double p) { return (k < 0 || k > n) ? 0.0 : std::exp(logpmf(k, n, p)); }
double binom::cdf(int k, int n, double p) {
  if (k < 0) return 0.0;
  if (k >= n) return 1.0;
  return d::betainc(n - k, k + 1, 1.0 - p);  // regularized incomplete beta (bdtr)
}
double binom::sf(int k, int n, double p) { return 1.0 - cdf(k, n, p); }
int binom::ppf(double q, int n, double p) {
  for (int k = 0; k <= n; ++k) if (cdf(k, n, p) >= q - 1e-12) return k;
  return n;
}
double binom::mean(int n, double p) { return n * p; }
double binom::var(int n, double p) { return n * p * (1 - p); }

// ---- poisson ----
double poisson::logpmf(int k, double mu) { return k < 0 ? -INFINITY : -mu + k * std::log(mu) - std::lgamma(k + 1.0); }
double poisson::pmf(int k, double mu) { return k < 0 ? 0.0 : std::exp(logpmf(k, mu)); }
double poisson::cdf(int k, double mu) { return k < 0 ? 0.0 : d::gammaincc(k + 1.0, mu); }  // pdtr
double poisson::sf(int k, double mu) { return k < 0 ? 1.0 : d::gammainc(k + 1.0, mu); }
int poisson::ppf(double q, double mu) {
  int k = 0;
  while (cdf(k, mu) < q - 1e-12 && k < 1000000) ++k;
  return k;
}
double poisson::mean(double mu) { return mu; }
double poisson::var(double mu) { return mu; }

// ---- geom (support k >= 1) ----
double geom::pmf(int k, double p) { return k < 1 ? 0.0 : std::pow(1 - p, k - 1) * p; }
double geom::cdf(int k, double p) { return k < 1 ? 0.0 : 1.0 - std::pow(1 - p, k); }
double geom::sf(int k, double p) { return k < 1 ? 1.0 : std::pow(1 - p, k); }
int geom::ppf(double q, double p) { return static_cast<int>(std::ceil(std::log1p(-q) / std::log1p(-p))); }
double geom::mean(double p) { return 1.0 / p; }
double geom::var(double p) { return (1 - p) / (p * p); }

// ---- bernoulli ----
double bernoulli::pmf(int k, double p) { return k == 1 ? p : (k == 0 ? 1 - p : 0.0); }
double bernoulli::cdf(int k, double p) { return k < 0 ? 0.0 : (k >= 1 ? 1.0 : 1 - p); }
double bernoulli::mean(double p) { return p; }
double bernoulli::var(double p) { return p * (1 - p); }

// ---- nbinom (number of failures before n successes) ----
double nbinom::logpmf(int k, double n, double p) {
  if (k < 0) return -INFINITY;
  return std::lgamma(k + n) - std::lgamma(k + 1.0) - std::lgamma(n) + n * std::log(p) + k * std::log1p(-p);
}
double nbinom::pmf(int k, double n, double p) { return k < 0 ? 0.0 : std::exp(logpmf(k, n, p)); }
double nbinom::cdf(int k, double n, double p) { return k < 0 ? 0.0 : d::betainc(n, k + 1, p); }
double nbinom::sf(int k, double n, double p) { return 1.0 - cdf(k, n, p); }
double nbinom::mean(double n, double p) { return n * (1 - p) / p; }
double nbinom::var(double n, double p) { return n * (1 - p) / (p * p); }

// ---- hypergeom (M total, n successes, N draws) ----
double hypergeom::pmf(int k, int M, int n, int N) {
  int lo = std::max(0, N - (M - n)), hi = std::min(n, N);
  if (k < lo || k > hi) return 0.0;
  return std::exp(lchoose(n, k) + lchoose(M - n, N - k) - lchoose(M, N));
}
double hypergeom::cdf(int k, int M, int n, int N) {
  double s = 0; for (int i = std::max(0, N - (M - n)); i <= k; ++i) s += pmf(i, M, n, N); return s;
}
double hypergeom::sf(int k, int M, int n, int N) { return 1.0 - cdf(k, M, n, N); }
double hypergeom::mean(int M, int n, int N) { return static_cast<double>(N) * n / M; }
double hypergeom::var(int M, int n, int N) {
  double Md = M, nd = n, Nd = N;
  return Nd * (nd / Md) * ((Md - nd) / Md) * ((Md - Nd) / (Md - 1));
}

}  // namespace scypp::stats
