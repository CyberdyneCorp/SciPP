// Filter design: IIR (butter/cheby1/cheby2 via analog prototype + bilinear),
// FIR (firwin), and representation conversions (tf2zpk/zpk2tf/zpk2sos/tf2sos).
#include "scypp/signal/signal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;
using cd = std::complex<double>;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct ZPK { std::vector<cd> z, p; double k; };

// Expand product (x - root_i) into ascending-power then return descending coeffs.
std::vector<cd> poly(const std::vector<cd>& roots) {
  std::vector<cd> c{1.0};
  for (cd r : roots) {
    std::vector<cd> nc(c.size() + 1, cd(0));
    for (size_t i = 0; i < c.size(); ++i) { nc[i] += c[i]; nc[i + 1] -= r * c[i]; }
    c = nc;
  }
  return c;  // descending powers
}

ndarray real_of(const std::vector<cd>& c) {
  std::vector<double> r(c.size());
  for (size_t i = 0; i < c.size(); ++i) r[i] = c[i].real();
  return sd::from_vec(r);
}

ZPK buttap(int N) {
  ZPK r; r.k = 1.0;
  for (int i = 0; i < N; ++i) {
    double m = -N + 1 + 2 * i;
    r.p.push_back(-std::exp(cd(0, kPi * m / (2.0 * N))));
  }
  return r;
}
ZPK cheb1ap(int N, double rp) {
  ZPK r;
  double eps = std::sqrt(std::pow(10.0, 0.1 * rp) - 1.0);
  double mu = std::asinh(1.0 / eps) / N;
  for (int i = 0; i < N; ++i) {
    double m = -N + 1 + 2 * i;
    double theta = kPi * m / (2.0 * N);
    r.p.push_back(-std::sinh(cd(mu, theta)));
  }
  cd prod = 1.0; for (cd pp : r.p) prod *= -pp;
  r.k = prod.real();
  if (N % 2 == 0) r.k /= std::sqrt(1.0 + eps * eps);
  return r;
}
ZPK cheb2ap(int N, double rs) {
  ZPK r;
  double de = 1.0 / std::sqrt(std::pow(10.0, 0.1 * rs) - 1.0);
  double mu = std::asinh(1.0 / de) / N;
  std::vector<double> mz;
  if (N % 2) { for (int m = -N + 1; m < 0; m += 2) mz.push_back(m); for (int m = 2; m < N; m += 2) mz.push_back(m); }
  else { for (int m = -N + 1; m < N; m += 2) mz.push_back(m); }
  for (double m : mz) r.z.push_back(-std::conj(cd(0, 1.0) / std::sin(cd(kPi * m / (2.0 * N), 0))));
  for (int i = 0; i < N; ++i) {
    double m = -N + 1 + 2 * i;
    cd pp = -std::exp(cd(0, kPi * m / (2.0 * N)));
    pp = cd(std::sinh(mu) * pp.real(), std::cosh(mu) * pp.imag());
    r.p.push_back(1.0 / pp);
  }
  cd np = 1.0, nz = 1.0;
  for (cd pp : r.p) np *= -pp;
  for (cd zz : r.z) nz *= -zz;
  r.k = (np / nz).real();
  return r;
}

int degree(const ZPK& f) { return static_cast<int>(f.p.size() - f.z.size()); }

ZPK lp2lp(ZPK f, double wo) {
  int d = degree(f);
  for (cd& z : f.z) z *= wo;
  for (cd& p : f.p) p *= wo;
  f.k *= std::pow(wo, d);
  return f;
}
ZPK lp2hp(ZPK f, double wo) {
  int d = degree(f);
  cd numz = 1, denp = 1;
  for (cd z : f.z) numz *= -z;
  for (cd p : f.p) denp *= -p;
  for (cd& z : f.z) z = wo / z;
  for (cd& p : f.p) p = wo / p;
  for (int i = 0; i < d; ++i) f.z.push_back(0.0);
  f.k *= (numz / denp).real();
  return f;
}
ZPK lp2bp(ZPK f, double wo, double bw) {
  int d = degree(f);
  std::vector<cd> z, p;
  for (cd zz : f.z) { cd s = zz * bw / 2.0, r = std::sqrt(s * s - wo * wo); z.push_back(s + r); z.push_back(s - r); }
  for (cd pp : f.p) { cd s = pp * bw / 2.0, r = std::sqrt(s * s - wo * wo); p.push_back(s + r); p.push_back(s - r); }
  for (int i = 0; i < d; ++i) z.push_back(0.0);
  f.z = z; f.p = p; f.k *= std::pow(bw, d);
  return f;
}
ZPK lp2bs(ZPK f, double wo, double bw) {
  int d = degree(f);
  cd numz = 1, denp = 1;
  for (cd z : f.z) numz *= -z;
  for (cd p : f.p) denp *= -p;
  std::vector<cd> z, p;
  for (cd zz : f.z) { cd s = (bw / 2.0) / zz, r = std::sqrt(s * s - wo * wo); z.push_back(s + r); z.push_back(s - r); }
  for (cd pp : f.p) { cd s = (bw / 2.0) / pp, r = std::sqrt(s * s - wo * wo); p.push_back(s + r); p.push_back(s - r); }
  for (int i = 0; i < d; ++i) { z.push_back(cd(0, wo)); z.push_back(cd(0, -wo)); }
  f.z = z; f.p = p; f.k *= (numz / denp).real();
  return f;
}
ZPK bilinear(ZPK f, double fs) {
  int d = degree(f);
  double fs2 = 2.0 * fs;
  cd numz = 1, denp = 1;
  for (cd z : f.z) numz *= (fs2 - z);
  for (cd p : f.p) denp *= (fs2 - p);
  for (cd& z : f.z) z = (fs2 + z) / (fs2 - z);
  for (cd& p : f.p) p = (fs2 + p) / (fs2 - p);
  for (int i = 0; i < d; ++i) f.z.push_back(-1.0);
  f.k *= (numz / denp).real();
  return f;
}

BA zpk2tf_internal(const ZPK& f) {
  std::vector<cd> b = poly(f.z), a = poly(f.p);
  for (cd& c : b) c *= f.k;
  return {real_of(b), real_of(a)};
}

BA iir_design(ZPK proto, const std::vector<double>& Wn, const std::string& btype) {
  double fs = 2.0;
  std::vector<double> warped;
  for (double w : Wn) warped.push_back(2.0 * fs * std::tan(kPi * w / fs));
  ZPK t;
  if (btype == "lowpass" || btype == "low") t = lp2lp(proto, warped[0]);
  else if (btype == "highpass" || btype == "high") t = lp2hp(proto, warped[0]);
  else {
    double bw = warped[1] - warped[0], wo = std::sqrt(warped[0] * warped[1]);
    if (btype == "bandpass" || btype == "band") t = lp2bp(proto, wo, bw);
    else if (btype == "bandstop" || btype == "stop") t = lp2bs(proto, wo, bw);
    else throw scypp::value_error("unknown btype " + btype);
  }
  return zpk2tf_internal(bilinear(t, fs));
}
}  // namespace

BA butter(int N, const std::vector<double>& Wn, const std::string& btype) {
  return iir_design(buttap(N), Wn, btype);
}
BA cheby1(int N, double rp, const std::vector<double>& Wn, const std::string& btype) {
  return iir_design(cheb1ap(N, rp), Wn, btype);
}
BA cheby2(int N, double rs, const std::vector<double>& Wn, const std::string& btype) {
  return iir_design(cheb2ap(N, rs), Wn, btype);
}

ndarray firwin(int numtaps, const std::vector<double>& cutoff, bool pass_zero,
               const std::string& window, double fs) {
  double nyq = fs / 2.0;
  std::vector<double> c;
  for (double x : cutoff) c.push_back(x / nyq);
  bool pass_nyquist = (c.size() & 1) ^ pass_zero;
  std::vector<double> edges;
  if (pass_zero) edges.push_back(0.0);
  for (double x : c) edges.push_back(x);
  if (pass_nyquist) edges.push_back(1.0);

  double alpha = 0.5 * (numtaps - 1);
  auto sinc = [](double x) { return x == 0.0 ? 1.0 : std::sin(kPi * x) / (kPi * x); };
  std::vector<double> h(numtaps, 0.0), m(numtaps);
  for (int i = 0; i < numtaps; ++i) m[i] = i - alpha;
  for (size_t bnd = 0; bnd + 1 < edges.size(); bnd += 2) {
    double l = edges[bnd], r = edges[bnd + 1];
    for (int i = 0; i < numtaps; ++i) h[i] += r * sinc(r * m[i]) - l * sinc(l * m[i]);
  }
  std::vector<double> win = sd::to_vec(get_window(window, numtaps, /*fftbins=*/false));
  for (int i = 0; i < numtaps; ++i) h[i] *= win[i];

  double l0 = edges[0], r0 = edges[1], sf;
  if (l0 == 0.0) sf = 0.0; else if (r0 == 1.0) sf = 1.0; else sf = 0.5 * (l0 + r0);
  double s = 0.0;
  for (int i = 0; i < numtaps; ++i) s += h[i] * std::cos(kPi * m[i] * sf);
  for (double& v : h) v /= s;
  return sd::from_vec(h);
}

namespace {
std::vector<cd> roots_of(const std::vector<double>& coeff) {
  // strip leading zeros, build companion, eigen-decompose.
  size_t start = 0; while (start < coeff.size() && coeff[start] == 0.0) ++start;
  std::vector<double> c(coeff.begin() + start, coeff.end());
  int n = static_cast<int>(c.size()) - 1;
  if (n <= 0) return {};
  std::vector<double> comp(n * n, 0.0);
  for (int j = 0; j < n; ++j) comp[0 * n + j] = -c[j + 1] / c[0];
  for (int i = 1; i < n; ++i) comp[i * n + (i - 1)] = 1.0;
  numpp::linalg::EigResult e = numpp::linalg::eig(sd::from_mat(comp, n, n));
  numpp::ndarray ev = e.eigenvalues.astype(numpp::kComplex128).ascontiguousarray();
  const cd* p = ev.typed_data<cd>();
  return std::vector<cd>(p, p + n);
}
numpp::ndarray complex_vec(const std::vector<cd>& v) {
  numpp::ndarray a(numpp::Shape{static_cast<int64_t>(v.size())}, numpp::kComplex128);
  cd* p = a.typed_data<cd>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}
}  // namespace

Zpk tf2zpk(const ndarray& b, const ndarray& a) {
  std::vector<double> bv = sd::to_vec(b), av = sd::to_vec(a);
  double k = bv[0] / av[0];
  return {complex_vec(roots_of(bv)), complex_vec(roots_of(av)), k};
}

BA zpk2tf(const ndarray& z, const ndarray& p, double k) {
  numpp::ndarray Z = z.astype(numpp::kComplex128).ascontiguousarray();
  numpp::ndarray P = p.astype(numpp::kComplex128).ascontiguousarray();
  const cd* zp = Z.typed_data<cd>();
  const cd* pp = P.typed_data<cd>();
  ZPK f;
  f.z.assign(zp, zp + Z.size());
  f.p.assign(pp, pp + P.size());
  f.k = k;
  return zpk2tf_internal(f);
}

namespace {
// Build second-order sections from conjugate-paired zpk (valid factorization;
// section ordering need not match scipy since the overall response is identical).
ndarray build_sos(ZPK f) {
  std::vector<cd> z = f.z, p = f.p;
  while (z.size() < p.size()) z.push_back(0.0);
  while (p.size() < z.size()) p.push_back(0.0);
  int n = static_cast<int>(p.size());
  int nsec = (n + 1) / 2;
  std::vector<double> sos(nsec * 6, 0.0);
  auto take_pair = [](std::vector<cd>& v, double& c1, double& c2) {
    cd a = v.front(); v.erase(v.begin());
    if (std::fabs(a.imag()) > 1e-12) {
      // remove conjugate
      for (size_t i = 0; i < v.size(); ++i)
        if (std::fabs(v[i].real() - a.real()) < 1e-9 && std::fabs(v[i].imag() + a.imag()) < 1e-9) { v.erase(v.begin() + i); break; }
      c1 = -2.0 * a.real(); c2 = std::norm(a);
    } else if (!v.empty()) {
      cd b = v.front(); v.erase(v.begin());
      c1 = -(a.real() + b.real()); c2 = a.real() * b.real();
    } else {
      c1 = -a.real(); c2 = 0.0;
    }
  };
  for (int s = 0; s < nsec; ++s) {
    double b1, b2, a1, a2;
    take_pair(z, b1, b2);
    take_pair(p, a1, a2);
    double* row = &sos[s * 6];
    row[0] = (s == 0 ? f.k : 1.0); row[1] = (s == 0 ? f.k : 1.0) * b1; row[2] = (s == 0 ? f.k : 1.0) * b2;
    row[3] = 1.0; row[4] = a1; row[5] = a2;
  }
  return sd::from_mat(sos, nsec, 6);
}
}  // namespace

ndarray zpk2sos(const ndarray& z, const ndarray& p, double k) {
  numpp::ndarray Z = z.astype(numpp::kComplex128).ascontiguousarray();
  numpp::ndarray P = p.astype(numpp::kComplex128).ascontiguousarray();
  const cd* zp = Z.typed_data<cd>();
  const cd* pp = P.typed_data<cd>();
  ZPK f; f.z.assign(zp, zp + Z.size()); f.p.assign(pp, pp + P.size()); f.k = k;
  return build_sos(f);
}

ndarray tf2sos(const ndarray& b, const ndarray& a) {
  Zpk zpk = tf2zpk(b, a);
  return zpk2sos(zpk.z, zpk.p, zpk.k);
}

}  // namespace scypp::signal
