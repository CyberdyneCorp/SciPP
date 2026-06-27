// Elliptic and Bessel filter design (analog prototypes + the shared IIR
// pipeline). Elliptic uses complete/incomplete elliptic integrals and Jacobi
// functions; Bessel uses reverse-Bessel-polynomial roots.
#include "scypp/signal/signal.hpp"

#include <cmath>
#include <complex>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scypp/linalg/detail.hpp"
#include "scypp/signal/iir_detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;
using iir::cd;
using iir::ZPK;
constexpr double kPi = 3.141592653589793238462643383279502884;

double ellipk(double m) {  // complete elliptic integral, first kind (param m)
  double a = 1.0, b = std::sqrt(1.0 - m);
  for (int i = 0; i < 100 && std::fabs(a - b) > 1e-15; ++i) { double an = 0.5 * (a + b); b = std::sqrt(a * b); a = an; }
  return kPi / (2.0 * a);
}
// Carlson symmetric integral R_F for the incomplete integral.
double rf(double x, double y, double z) {
  for (int i = 0; i < 100; ++i) {
    double l = std::sqrt(x) * std::sqrt(y) + std::sqrt(y) * std::sqrt(z) + std::sqrt(z) * std::sqrt(x);
    x = 0.25 * (x + l); y = 0.25 * (y + l); z = 0.25 * (z + l);
    double mu = (x + y + z) / 3.0;
    if (std::fabs(x - mu) + std::fabs(y - mu) + std::fabs(z - mu) < 1e-14 * mu) break;
  }
  double mu = (x + y + z) / 3.0;
  return 1.0 / std::sqrt(mu);
}
double ellipkinc(double phi, double m) {  // incomplete elliptic integral F(phi|m)
  double s = std::sin(phi), c = std::cos(phi);
  return s * rf(c * c, 1.0 - m * s * s, 1.0);
}
void ellipj(double u, double m, double& sn, double& cn, double& dn) {
  if (m < 1e-15) { sn = std::sin(u); cn = std::cos(u); dn = 1.0; return; }
  double a[20], c[20], b = std::sqrt(1.0 - m);
  a[0] = 1.0; c[0] = std::sqrt(m);
  int n = 0;
  while (std::fabs(c[n]) > 1e-15 && n < 18) { a[n + 1] = 0.5 * (a[n] + b); c[n + 1] = 0.5 * (a[n] - b); b = std::sqrt(a[n] * b); ++n; }
  double phi = std::ldexp(a[n] * u, n);
  for (int i = n; i > 0; --i) phi = 0.5 * (phi + std::asin(c[i] / a[i] * std::sin(phi)));
  sn = std::sin(phi); cn = std::cos(phi); dn = std::sqrt(1.0 - m * sn * sn);
}

double solve_m(double krat) {  // K(m)/K(1-m) = krat, m in (0,1), monotone increasing
  double lo = 1e-12, hi = 1.0 - 1e-12;
  for (int i = 0; i < 200; ++i) {
    double mid = 0.5 * (lo + hi);
    if (ellipk(mid) / ellipk(1.0 - mid) < krat) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

ZPK ellipap(int N, double rp, double rs) {
  ZPK out;
  double eps_sq = std::pow(10.0, 0.1 * rp) - 1.0, eps = std::sqrt(eps_sq);
  if (N == 1) { double p = -std::sqrt(1.0 / eps_sq); out.p = {cd(p, 0)}; out.k = -p; return out; }
  double ck1_sq = eps_sq / (std::pow(10.0, 0.1 * rs) - 1.0);
  double krat = N * ellipk(ck1_sq) / ellipk(1.0 - ck1_sq);
  double m = solve_m(krat);
  double capk = ellipk(m);
  std::vector<cd> z;
  for (int jj = 1 - N % 2; jj < N; jj += 2) {
    double s, c, d; ellipj(jj * capk / N, m, s, c, d);
    if (std::fabs(s) > 1e-12) { cd zz = cd(0, 1) * (1.0 / (std::sqrt(m) * s)); z.push_back(zz); z.push_back(std::conj(zz)); }
  }
  double v0 = capk * ellipkinc(std::atan(1.0 / eps), 1.0 - m) / (N * ellipk(ck1_sq));
  double sv, cv, dv; ellipj(v0, 1.0 - m, sv, cv, dv);
  std::vector<cd> p;
  for (int jj = 1 - N % 2; jj < N; jj += 2) {
    double s, c, d; ellipj(jj * capk / N, m, s, c, d);
    cd pp = -(cd(c * d * sv * cv, s * dv)) / (1.0 - (d * sv) * (d * sv));
    if (std::fabs(pp.imag()) > 1e-12) { p.push_back(pp); p.push_back(std::conj(pp)); }
    else p.push_back(pp);
  }
  out.z = z; out.p = p;
  cd np = 1, nz = 1;
  for (cd pp : p) np *= -pp;
  for (cd zz : z) nz *= -zz;
  double k = (np / nz).real();
  if (N % 2 == 0) k /= std::sqrt(1.0 + eps_sq);
  out.k = k;
  return out;
}

double lfact(int n) { double s = 0; for (int i = 2; i <= n; ++i) s += std::log(i); return s; }

ZPK besselap(int N) {
  // Reverse Bessel polynomial, descending: coeff of s^{N-k} = (N+k)!/((N-k)! k! 2^k).
  std::vector<double> b(N + 1);
  for (int k = 0; k <= N; ++k)
    b[k] = std::exp(lfact(N + k) - lfact(N - k) - lfact(k) - k * std::log(2.0));
  std::vector<double> comp(N * N, 0.0);
  for (int j = 0; j < N; ++j) comp[0 * N + j] = -b[j + 1] / b[0];
  for (int i = 1; i < N; ++i) comp[i * N + (i - 1)] = 1.0;
  numpp::linalg::EigResult e = numpp::linalg::eig(sd::from_mat(comp, N, N));
  numpp::ndarray ev = e.eigenvalues.astype(numpp::kComplex128).ascontiguousarray();
  const cd* pp = ev.typed_data<cd>();
  ZPK out;
  out.p.assign(pp, pp + N);
  double scale = std::pow(b[N], -1.0 / N);  // 'phase' normalization
  for (cd& p : out.p) p *= scale;           // prod(-poles) becomes 1 -> DC gain 1
  out.k = 1.0;
  return out;
}
}  // namespace

BA ellip(int N, double rp, double rs, const std::vector<double>& Wn, const std::string& btype) {
  iir::BAcoef c = iir::design(ellipap(N, rp, rs), Wn, btype);
  return {c.b, c.a};
}
BA bessel(int N, const std::vector<double>& Wn, const std::string& btype) {
  iir::BAcoef c = iir::design(besselap(N), Wn, btype);
  return {c.b, c.a};
}

}  // namespace scypp::signal
