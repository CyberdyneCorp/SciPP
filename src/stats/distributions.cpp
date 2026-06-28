// Continuous distributions: norm, expon, uniform, gamma, chi2, beta, t, f.
#include "scipp/stats/stats.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scipp/special/special.hpp"
#include "scipp/stats/detail.hpp"

namespace scipp::stats {
namespace {
constexpr double kSqrt2 = 1.4142135623730951;
constexpr double kSqrt2Pi = 2.5066282746310002;
constexpr double kLog2Pi = 1.8378770664093453;
namespace d = detail;

template <class F>
ndarray map_x(const ndarray& x, F&& f) {
  numpp::ndarray xc = x.astype(numpp::kFloat64).ascontiguousarray();
  numpp::ndarray out(xc.shape(), numpp::kFloat64);
  const double* in = xc.typed_data<double>();
  double* o = out.typed_data<double>();
  for (int64_t i = 0; i < xc.size(); ++i) o[i] = f(in[i]);
  return out;
}
}  // namespace

// ---- norm ----
double norm::pdf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return std::exp(-0.5 * z * z) / (kSqrt2Pi * scale);
}
double norm::logpdf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return -0.5 * z * z - 0.5 * kLog2Pi - std::log(scale);
}
double norm::cdf(double x, double loc, double scale) {
  return 0.5 * std::erfc(-(x - loc) / scale / kSqrt2);
}
double norm::sf(double x, double loc, double scale) {
  return 0.5 * std::erfc((x - loc) / scale / kSqrt2);
}
double norm::ppf(double q, double loc, double scale) {
  return loc + scale * kSqrt2 * special::erfinv(2.0 * q - 1.0);
}
double norm::mean(double loc, double) { return loc; }
double norm::var(double, double scale) { return scale * scale; }
double norm::std(double, double scale) { return scale; }
double norm::median(double loc, double) { return loc; }
ndarray norm::pdf(const ndarray& x, double loc, double scale) {
  return map_x(x, [=](double v) { return pdf(v, loc, scale); });
}
ndarray norm::cdf(const ndarray& x, double loc, double scale) {
  return map_x(x, [=](double v) { return cdf(v, loc, scale); });
}

// ---- expon ----
double expon::pdf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return z < 0 ? 0.0 : std::exp(-z) / scale;
}
double expon::logpdf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return z < 0 ? -INFINITY : -z - std::log(scale);
}
double expon::cdf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return z < 0 ? 0.0 : 1.0 - std::exp(-z);
}
double expon::sf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return z < 0 ? 1.0 : std::exp(-z);
}
double expon::ppf(double q, double loc, double scale) { return loc - scale * std::log1p(-q); }
double expon::mean(double loc, double scale) { return loc + scale; }
double expon::var(double, double scale) { return scale * scale; }
double expon::median(double loc, double scale) { return loc + scale * 0.6931471805599453; }

// ---- uniform ----
double uniform::pdf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return (z < 0 || z > 1) ? 0.0 : 1.0 / scale;
}
double uniform::cdf(double x, double loc, double scale) {
  double z = (x - loc) / scale;
  return z < 0 ? 0.0 : (z > 1 ? 1.0 : z);
}
double uniform::sf(double x, double loc, double scale) { return 1.0 - cdf(x, loc, scale); }
double uniform::ppf(double q, double loc, double scale) { return loc + scale * q; }
double uniform::mean(double loc, double scale) { return loc + 0.5 * scale; }
double uniform::var(double, double scale) { return scale * scale / 12.0; }

// ---- gamma ----
double gamma::pdf(double x, double a, double loc, double scale) {
  double z = (x - loc) / scale;
  if (z <= 0) return 0.0;
  return std::exp((a - 1.0) * std::log(z) - z - std::lgamma(a)) / scale;
}
double gamma::logpdf(double x, double a, double loc, double scale) {
  double z = (x - loc) / scale;
  if (z <= 0) return -INFINITY;
  return (a - 1.0) * std::log(z) - z - std::lgamma(a) - std::log(scale);
}
double gamma::cdf(double x, double a, double loc, double scale) {
  double z = (x - loc) / scale;
  return z <= 0 ? 0.0 : d::gammainc(a, z);
}
double gamma::sf(double x, double a, double loc, double scale) {
  double z = (x - loc) / scale;
  return z <= 0 ? 1.0 : d::gammaincc(a, z);
}
double gamma::ppf(double q, double a, double loc, double scale) {
  return loc + scale * d::gammaincinv(a, q);
}
double gamma::mean(double a, double loc, double scale) { return loc + scale * a; }
double gamma::var(double a, double, double scale) { return scale * scale * a; }

// ---- chi2 (= gamma(df/2, scale=2)) ----
double chi2::pdf(double x, double df, double loc, double scale) {
  return gamma::pdf(x, df / 2.0, loc, 2.0 * scale);
}
double chi2::cdf(double x, double df, double loc, double scale) {
  return gamma::cdf(x, df / 2.0, loc, 2.0 * scale);
}
double chi2::sf(double x, double df, double loc, double scale) {
  return gamma::sf(x, df / 2.0, loc, 2.0 * scale);
}
double chi2::ppf(double q, double df, double loc, double scale) {
  return gamma::ppf(q, df / 2.0, loc, 2.0 * scale);
}
double chi2::mean(double df, double loc, double scale) { return loc + df * scale; }
double chi2::var(double df, double, double scale) { return 2.0 * df * scale * scale; }

// ---- beta ----
double beta::pdf(double x, double a, double b, double loc, double scale) {
  double z = (x - loc) / scale;
  if (z <= 0 || z >= 1) return 0.0;
  double lb = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  return std::exp((a - 1.0) * std::log(z) + (b - 1.0) * std::log(1.0 - z) - lb) / scale;
}
double beta::cdf(double x, double a, double b, double loc, double scale) {
  double z = (x - loc) / scale;
  return d::betainc(a, b, z);
}
double beta::sf(double x, double a, double b, double loc, double scale) {
  return 1.0 - cdf(x, a, b, loc, scale);
}
double beta::ppf(double q, double a, double b, double loc, double scale) {
  return loc + scale * d::betaincinv(a, b, q);
}
double beta::mean(double a, double b, double loc, double scale) { return loc + scale * a / (a + b); }
double beta::var(double a, double b, double, double scale) {
  return scale * scale * a * b / ((a + b) * (a + b) * (a + b + 1.0));
}

// ---- t ----
double t::pdf(double x, double df, double loc, double scale) {
  double z = (x - loc) / scale;
  double lp = std::lgamma(0.5 * (df + 1.0)) - std::lgamma(0.5 * df) - 0.5 * std::log(df * M_PI) -
              0.5 * (df + 1.0) * std::log1p(z * z / df);
  return std::exp(lp) / scale;
}
double t::cdf(double x, double df, double loc, double scale) {
  double z = (x - loc) / scale;
  double xt = df / (df + z * z);
  double ib = d::betainc(0.5 * df, 0.5, xt);
  return z > 0 ? 1.0 - 0.5 * ib : 0.5 * ib;
}
double t::sf(double x, double df, double loc, double scale) { return 1.0 - cdf(x, df, loc, scale); }
double t::ppf(double q, double df, double loc, double scale) {
  double z;
  if (q == 0.5) z = 0.0;
  else if (q < 0.5) { double xt = d::betaincinv(0.5 * df, 0.5, 2.0 * q); z = -std::sqrt(df * (1.0 / xt - 1.0)); }
  else { double xt = d::betaincinv(0.5 * df, 0.5, 2.0 * (1.0 - q)); z = std::sqrt(df * (1.0 / xt - 1.0)); }
  return loc + scale * z;
}
double t::mean(double, double loc, double) { return loc; }
double t::var(double df, double, double scale) { return scale * scale * df / (df - 2.0); }

// ---- f ----
double f::pdf(double x, double dfn, double dfd, double loc, double scale) {
  double z = (x - loc) / scale;
  if (z <= 0) return 0.0;
  double lb = std::lgamma(0.5 * dfn) + std::lgamma(0.5 * dfd) - std::lgamma(0.5 * (dfn + dfd));
  double lp = 0.5 * dfn * std::log(dfn) + 0.5 * dfd * std::log(dfd) + (0.5 * dfn - 1.0) * std::log(z) -
              0.5 * (dfn + dfd) * std::log(dfd + dfn * z) - lb;
  return std::exp(lp) / scale;
}
double f::cdf(double x, double dfn, double dfd, double loc, double scale) {
  double z = (x - loc) / scale;
  if (z <= 0) return 0.0;
  return d::betainc(0.5 * dfn, 0.5 * dfd, dfn * z / (dfn * z + dfd));
}
double f::sf(double x, double dfn, double dfd, double loc, double scale) {
  return 1.0 - cdf(x, dfn, dfd, loc, scale);
}
double f::ppf(double q, double dfn, double dfd, double loc, double scale) {
  double w = d::betaincinv(0.5 * dfn, 0.5 * dfd, q);
  return loc + scale * dfd * w / (dfn * (1.0 - w));
}
double f::mean(double, double dfd, double loc, double scale) {
  return loc + scale * dfd / (dfd - 2.0);
}

}  // namespace scipp::stats
