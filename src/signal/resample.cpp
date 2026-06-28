// Resampling: resample (FFT), upfirdn, resample_poly, decimate.
#include "scipp/signal/signal.hpp"

#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/fft/fft.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::signal {
namespace {
namespace sd = scipp::linalg::detail;
using cd = std::complex<double>;
constexpr double kPi = 3.141592653589793238462643383279502884;

std::vector<cd> to_cplx(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kComplex128).ascontiguousarray();
  const cd* p = c.typed_data<cd>();
  return std::vector<cd>(p, p + c.size());
}
numpp::ndarray from_cplx(const std::vector<cd>& v) {
  numpp::ndarray a(numpp::Shape{static_cast<int64_t>(v.size())}, numpp::kComplex128);
  cd* p = a.typed_data<cd>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}

std::vector<double> full_conv(const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> out(a.size() + b.size() - 1, 0.0);
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < b.size(); ++j) out[i + j] += a[i] * b[j];
  return out;
}
}  // namespace

ndarray resample(const ndarray& x, int64_t num) {
  std::vector<double> xv = sd::to_vec(x);
  int Nx = static_cast<int>(xv.size());
  std::vector<cd> X = to_cplx(numpp::fft::fft(x));
  std::vector<cd> Y(num, cd(0));
  int N = std::min<int>(num, Nx);
  int nyq = N / 2 + 1;
  for (int i = 0; i < nyq; ++i) Y[i] = X[i];
  int nneg = (N - 1) / 2;
  for (int i = 1; i <= nneg; ++i) Y[num - i] = X[Nx - i];
  if (N % 2 == 0) {
    if (num < Nx) Y[N / 2] += X[Nx - N / 2];
    else if (num > Nx) { Y[N / 2] *= 0.5; Y[num - N / 2] = Y[N / 2]; }
  }
  std::vector<cd> y = to_cplx(numpp::fft::ifft(from_cplx(Y)));
  double scale = static_cast<double>(num) / Nx;
  std::vector<double> out(num);
  for (int64_t i = 0; i < num; ++i) out[i] = y[i].real() * scale;
  return sd::from_vec(out);
}

ndarray upfirdn(const ndarray& h, const ndarray& x, int up, int down) {
  std::vector<double> hv = sd::to_vec(h), xv = sd::to_vec(x);
  int n = static_cast<int>(xv.size());
  std::vector<double> xup((n - 1) * up + 1, 0.0);
  for (int i = 0; i < n; ++i) xup[i * up] = xv[i];
  std::vector<double> y = full_conv(xup, hv);
  std::vector<double> out;
  for (size_t i = 0; i < y.size(); i += down) out.push_back(y[i]);
  return sd::from_vec(out);
}

namespace {
// firwin(numtaps, fc, window=('kaiser', beta)) normalized, used by resample_poly.
std::vector<double> firwin_kaiser(int numtaps, double fc, double beta) {
  double alpha = 0.5 * (numtaps - 1);
  std::vector<double> win = sd::to_vec(kaiser(numtaps, beta, /*sym=*/true));
  std::vector<double> h(numtaps);
  auto sinc = [](double v) { return v == 0.0 ? 1.0 : std::sin(kPi * v) / (kPi * v); };
  double s = 0;
  for (int i = 0; i < numtaps; ++i) { double m = i - alpha; h[i] = fc * sinc(fc * m) * win[i]; s += h[i]; }
  for (double& v : h) v /= s;
  return h;
}
int output_len(int len_h, int len_x, int up, int down) {
  return ((len_x - 1) * up + len_h - 1) / down + 1;
}
}  // namespace

ndarray resample_poly(const ndarray& x, int up, int down) {
  std::vector<double> xv = sd::to_vec(x);
  int g = std::gcd(up, down);
  up /= g; down /= g;
  if (up == 1 && down == 1) return sd::from_vec(xv);
  int n = static_cast<int>(xv.size());
  int max_rate = std::max(up, down);
  double f_c = 1.0 / max_rate;
  int half_len = 10 * max_rate;
  std::vector<double> h = firwin_kaiser(2 * half_len + 1, f_c, 5.0);
  for (double& v : h) v *= up;

  int n_out = (n * up + down - 1) / down;  // ceil(n*up/down)
  int n_pre_pad = down - half_len % down;
  int n_post_pad = 0;
  int n_pre_remove = (half_len + n_pre_pad) / down;
  while (output_len(static_cast<int>(h.size()) + n_pre_pad + n_post_pad, n, up, down) <
         n_out + n_pre_remove)
    ++n_post_pad;
  std::vector<double> hpad;
  hpad.insert(hpad.end(), n_pre_pad, 0.0);
  hpad.insert(hpad.end(), h.begin(), h.end());
  hpad.insert(hpad.end(), n_post_pad, 0.0);
  std::vector<double> y = sd::to_vec(upfirdn(sd::from_vec(hpad), x, up, down));
  std::vector<double> out(y.begin() + n_pre_remove, y.begin() + n_pre_remove + n_out);
  return sd::from_vec(out);
}

ndarray decimate(const ndarray& x, int q) {
  BA f = cheby1(8, 0.05, {0.8 / q}, "low");
  std::vector<double> y = sd::to_vec(filtfilt(f.b, f.a, x));
  std::vector<double> out;
  for (size_t i = 0; i < y.size(); i += q) out.push_back(y[i]);
  return sd::from_vec(out);
}

}  // namespace scipp::signal
