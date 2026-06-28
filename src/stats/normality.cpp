// Normality tests: shapiro (Shapiro-Wilk, Royston AS R94) and anderson
// (Anderson-Darling for the normal distribution), matching scipy.stats.
#include "scipp/stats/stats.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "scipp/linalg/detail.hpp"

namespace scipp::stats {
namespace {
namespace sd = scipp::linalg::detail;

// Algorithm AS 181.2 polynomial evaluation: cc[0] + cc[1]*x + ... + cc[k-1]*x^(k-1).
double poly(const double* cc, int nord, double x) {
  double ret = cc[0];
  if (nord > 1) {
    double p = x * cc[nord - 1];
    for (int j = nord - 2; j > 0; --j) p = (p + cc[j]) * x;
    ret += p;
  }
  return ret;
}

// Shapiro-Wilk W statistic and p-value (Royston, AS R94, Appl. Statist. 1995).
// x must be sorted ascending. Returns {W, p}.
ShapiroResult swilk(std::vector<double>& x) {
  const int n = static_cast<int>(x.size());
  const int nn2 = n / 2;
  const double small = 1e-19;

  static const double g[2] = {-2.273, .459};
  static const double c1[6] = {0., .221157, -.147981, -2.071190, 4.434685, -2.706056};
  static const double c2[6] = {0., .042981, -.293762, -1.752461, 5.682633, -3.582633};
  static const double c3[4] = {.544, -.39978, .025054, -6.714e-4};
  static const double c4[4] = {1.3822, -.77857, .062767, -.0020322};
  static const double c5[4] = {-1.5861, -.31082, -.083751, .0038915};
  static const double c6[3] = {-.4803, -.082676, .0030302};

  std::vector<double> a(nn2 + 1, 0.0);  // 1-indexed half-weights
  const double an = n;

  if (n == 3) {
    a[1] = std::sqrt(0.5);
  } else {
    const double an25 = an + .25;
    double summ2 = 0.0;
    for (int i = 1; i <= nn2; ++i) {
      a[i] = norm::ppf((i - .375) / an25);
      summ2 += a[i] * a[i];
    }
    summ2 *= 2.0;
    const double ssumm2 = std::sqrt(summ2);
    const double rsn = 1.0 / std::sqrt(an);
    const double a1 = poly(c1, 6, rsn) - a[1] / ssumm2;

    int i1;
    double fac;
    if (n > 5) {
      i1 = 3;
      const double a2 = -a[2] / ssumm2 + poly(c2, 6, rsn);
      fac = std::sqrt((summ2 - 2.0 * a[1] * a[1] - 2.0 * a[2] * a[2]) /
                      (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2));
      a[2] = a2;
    } else {
      i1 = 2;
      fac = std::sqrt((summ2 - 2.0 * a[1] * a[1]) / (1.0 - 2.0 * a1 * a1));
    }
    a[1] = a1;
    for (int i = i1; i <= nn2; ++i) a[i] /= -fac;
  }

  const double range = x[n - 1] - x[0];
  if (range < small) throw std::invalid_argument("shapiro: data has zero range");

  // W statistic as squared correlation between data and normalised weights.
  double sa = 0.0, sx = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = n - 1 - i;
    double w;
    if (i < j) w = -a[i + 1];
    else if (i > j) w = a[j + 1];
    else w = 0.0;
    sa += w;
    sx += x[i] / range;
  }
  sa /= n;
  sx /= n;

  double ssa = 0.0, ssx = 0.0, sax = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = n - 1 - i;
    double w;
    if (i < j) w = -a[i + 1];
    else if (i > j) w = a[j + 1];
    else w = 0.0;
    const double asa = w - sa;
    const double xsx = x[i] / range - sx;
    ssa += asa * asa;
    ssx += xsx * xsx;
    sax += asa * xsx;
  }

  const double ssassx = std::sqrt(ssa * ssx);
  const double w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
  const double W = 1.0 - w1;

  // Significance level.
  double pw;
  if (n == 3) {
    const double pi6 = 1.90985931710274;     // 6/pi
    const double stqr = 1.04719755119660;    // asin(sqrt(3/4))
    pw = pi6 * (std::asin(std::sqrt(W)) - stqr);
    if (pw < 0.0) pw = 0.0;
    if (pw > 1.0) pw = 1.0;
    return {W, pw};
  }

  double y = std::log(w1);
  const double xx = std::log(an);
  double m, s;
  if (n <= 11) {
    const double gamma = poly(g, 2, an);
    if (y >= gamma) return {W, 1e-99};
    y = -std::log(gamma - y);
    m = poly(c3, 4, an);
    s = std::exp(poly(c4, 4, an));
  } else {
    m = poly(c5, 4, xx);
    s = std::exp(poly(c6, 3, xx));
  }
  pw = norm::sf((y - m) / s);
  return {W, pw};
}
}  // namespace

ShapiroResult shapiro(const ndarray& x) {
  auto v = sd::to_vec(x);
  if (v.size() < 3) throw std::invalid_argument("shapiro: need at least 3 observations");
  std::sort(v.begin(), v.end());
  return swilk(v);
}

AndersonResult anderson(const ndarray& x) {
  auto v = sd::to_vec(x);
  const int N = static_cast<int>(v.size());
  if (N < 2) throw std::invalid_argument("anderson: need at least 2 observations");
  std::sort(v.begin(), v.end());

  double mean = 0.0;
  for (double xi : v) mean += xi;
  mean /= N;
  double ss = 0.0;
  for (double xi : v) ss += (xi - mean) * (xi - mean);
  const double sdv = std::sqrt(ss / (N - 1));  // ddof=1

  // A^2 = -N - sum_{i=1..N} (2i-1)/N * (logcdf(w_i) + logsf(w_{N+1-i}))
  double A2 = -static_cast<double>(N);
  for (int i = 1; i <= N; ++i) {
    const double wi = (v[i - 1] - mean) / sdv;
    const double wr = (v[N - i] - mean) / sdv;
    const double logcdf = std::log(norm::cdf(wi));
    const double logsf = std::log(norm::sf(wr));
    A2 -= (2.0 * i - 1.0) / N * (logcdf + logsf);
  }

  // scipy 1.15: statistic is the raw A^2; the small-sample correction
  // (1 + 4/N - 25/N^2) is applied to the critical values instead.
  static const double avals[5] = {0.576, 0.656, 0.787, 0.918, 1.092};
  const double factor = 1.0 + 4.0 / N - 25.0 / (static_cast<double>(N) * N);
  std::vector<double> critical(5), sig = {15.0, 10.0, 5.0, 2.5, 1.0};
  for (int k = 0; k < 5; ++k)
    critical[k] = std::round(avals[k] / factor * 1000.0) / 1000.0;

  return {A2, critical, sig};
}

}  // namespace scipp::stats
