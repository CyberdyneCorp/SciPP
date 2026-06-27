// Time-domain filtering: lfilter, lfilter_zi, filtfilt, sosfilt, detrend,
// hilbert (analytic signal), freqz.
#include "scypp/signal/signal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/fft/fft.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;
using cd = std::complex<double>;
constexpr double kPi = 3.141592653589793238462643383279502884;

// DF2T difference equation with optional initial state zi (length n-1).
std::vector<double> lfilter_impl(std::vector<double> b, std::vector<double> a,
                                 const std::vector<double>& x, std::vector<double> zi) {
  double a0 = a[0];
  for (double& v : b) v /= a0;
  for (double& v : a) v /= a0;
  int n = std::max(a.size(), b.size());
  b.resize(n, 0.0);
  a.resize(n, 0.0);
  std::vector<double> z(n - 1, 0.0);
  if (!zi.empty()) z = zi;
  std::vector<double> y(x.size());
  for (size_t m = 0; m < x.size(); ++m) {
    double xm = x[m];
    double ym = b[0] * xm + (n > 1 ? z[0] : 0.0);
    for (int i = 1; i < n - 1; ++i) z[i - 1] = b[i] * xm + z[i] - a[i] * ym;
    if (n > 1) z[n - 2] = b[n - 1] * xm - a[n - 1] * ym;
    y[m] = ym;
  }
  return y;
}

std::vector<double> compute_zi(std::vector<double> b, std::vector<double> a) {
  double a0 = a[0];
  for (double& v : b) v /= a0;
  for (double& v : a) v /= a0;
  int n = std::max(a.size(), b.size());
  b.resize(n, 0.0);
  a.resize(n, 0.0);
  std::vector<double> B(n - 1), zi(n - 1, 0.0);
  double colsum = 1.0, bsum = 0.0;
  for (int k = 1; k < n; ++k) { B[k - 1] = b[k] - a[k] * b[0]; colsum += a[k]; bsum += B[k - 1]; }
  zi[0] = bsum / colsum;
  double asum = 1.0, csum = 0.0;
  for (int k = 1; k < n - 1; ++k) { asum += a[k]; csum += B[k - 1]; zi[k] = asum * zi[0] - csum; }
  return zi;
}
}  // namespace

ndarray lfilter(const ndarray& b, const ndarray& a, const ndarray& x) {
  return sd::from_vec(lfilter_impl(sd::to_vec(b), sd::to_vec(a), sd::to_vec(x), {}));
}

ndarray lfilter_zi(const ndarray& b, const ndarray& a) {
  return sd::from_vec(compute_zi(sd::to_vec(b), sd::to_vec(a)));
}

ndarray filtfilt(const ndarray& b, const ndarray& a, const ndarray& x) {
  std::vector<double> bv = sd::to_vec(b), av = sd::to_vec(a), xv = sd::to_vec(x);
  int ntaps = std::max(av.size(), bv.size());
  int edge = 3 * ntaps;
  if (edge >= static_cast<int>(xv.size())) throw scypp::value_error("filtfilt: input too short");
  // odd extension
  std::vector<double> ext;
  for (int i = edge; i >= 1; --i) ext.push_back(2.0 * xv.front() - xv[i]);
  ext.insert(ext.end(), xv.begin(), xv.end());
  int n = static_cast<int>(xv.size());
  for (int i = 2; i <= edge + 1; ++i) ext.push_back(2.0 * xv.back() - xv[n - i]);

  std::vector<double> zi = compute_zi(bv, av);
  auto scale = [&](double s) { std::vector<double> z = zi; for (double& v : z) v *= s; return z; };
  std::vector<double> y = lfilter_impl(bv, av, ext, scale(ext.front()));
  std::reverse(y.begin(), y.end());
  y = lfilter_impl(bv, av, y, scale(y.front()));
  std::reverse(y.begin(), y.end());
  return sd::from_vec(std::vector<double>(y.begin() + edge, y.end() - edge));
}

ndarray sosfilt(const ndarray& sos, const ndarray& x) {
  numpp::ndarray S = sos.astype(numpp::kFloat64).ascontiguousarray();
  int nsec = static_cast<int>(S.shape()[0]);
  const double* s = S.typed_data<double>();
  std::vector<double> y = sd::to_vec(x);
  for (int sec = 0; sec < nsec; ++sec) {
    const double* c = s + sec * 6;
    double a0 = c[3];
    double b0 = c[0] / a0, b1 = c[1] / a0, b2 = c[2] / a0, a1 = c[4] / a0, a2 = c[5] / a0;
    double z0 = 0.0, z1 = 0.0;
    for (double& xm : y) {
      double out = b0 * xm + z0;
      z0 = b1 * xm - a1 * out + z1;
      z1 = b2 * xm - a2 * out;
      xm = out;
    }
  }
  return sd::from_vec(y);
}

ndarray detrend(const ndarray& x, const std::string& type) {
  std::vector<double> v = sd::to_vec(x);
  int n = static_cast<int>(v.size());
  if (type == "constant") {
    double m = 0; for (double e : v) m += e; m /= n;
    for (double& e : v) e -= m;
  } else if (type == "linear") {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; ++i) { sx += i; sy += v[i]; sxx += static_cast<double>(i) * i; sxy += static_cast<double>(i) * v[i]; }
    double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    double intercept = (sy - slope * sx) / n;
    for (int i = 0; i < n; ++i) v[i] -= slope * i + intercept;
  } else {
    throw scypp::value_error("detrend: unknown type " + type);
  }
  return sd::from_vec(v);
}

ndarray hilbert(const ndarray& x) {
  numpp::ndarray Xf = numpp::fft::fft(x);
  numpp::ndarray Xc = Xf.astype(numpp::kComplex128).ascontiguousarray();
  int64_t N = Xc.size();
  cd* p = Xc.typed_data<cd>();
  std::vector<double> h(N, 0.0);
  if (N % 2 == 0) { h[0] = 1; h[N / 2] = 1; for (int64_t i = 1; i < N / 2; ++i) h[i] = 2; }
  else { h[0] = 1; for (int64_t i = 1; i <= (N - 1) / 2; ++i) h[i] = 2; }
  for (int64_t i = 0; i < N; ++i) p[i] *= h[i];
  return numpp::fft::ifft(Xc);
}

FreqzResult freqz(const ndarray& b, const ndarray& a, int64_t worN) {
  std::vector<double> bv = sd::to_vec(b), av = sd::to_vec(a);
  std::vector<double> w(worN);
  numpp::ndarray H(numpp::Shape{worN}, numpp::kComplex128);
  cd* h = H.typed_data<cd>();
  for (int64_t i = 0; i < worN; ++i) {
    double wi = kPi * i / worN;  // linspace(0, pi, worN, endpoint=False)
    w[i] = wi;
    cd num = 0, den = 0, e = std::exp(cd(0, -wi));
    cd ek = 1.0;
    for (double bk : bv) { num += bk * ek; ek *= e; }
    ek = 1.0;
    for (double ak : av) { den += ak * ek; ek *= e; }
    h[i] = num / den;
  }
  return {sd::from_vec(w), H};
}

}  // namespace scypp::signal
