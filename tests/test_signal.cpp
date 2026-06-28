// Oracle tests for scipp::signal against frozen SciPy golden data.
#include <cmath>
#include <complex>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/signal/signal.hpp"
#include "scipp_test.hpp"

namespace sg = scipp::signal;

namespace {
constexpr double R = 1e-7, A = 1e-9;
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
void cv(const numpp::ndarray& got, const double* exp, int n, double rtol = R, double atol = A) {
  auto g = tov(got);
  CHECK(static_cast<int>(g.size()) == n);
  for (int i = 0; i < n && i < (int)g.size(); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace
#define V(name) vec(golden::name, golden::name##_n)
#define G(name) golden::name, golden::name##_n

TEST_CASE("convolution") {
  auto a = V(sg_a), b = V(sg_b);
  cv(sg::convolve(a, b, "full"), G(sg_conv_full));
  cv(sg::convolve(a, b, "same"), G(sg_conv_same));
  cv(sg::convolve(a, b, "valid"), G(sg_conv_valid));
  cv(sg::correlate(a, b, "full"), G(sg_corr_full));
  cv(sg::fftconvolve(a, b, "full"), G(sg_conv_full));
}

TEST_CASE("windows") {
  cv(sg::hann(16), G(sg_hann));
  cv(sg::hamming(16), G(sg_hamming));
  cv(sg::blackman(16), G(sg_blackman));
  cv(sg::kaiser(16, 8.0), G(sg_kaiser));
  cv(sg::tukey(16, 0.5), G(sg_tukey));
  cv(sg::hann(16, false), G(sg_hann_per));
  cv(sg::get_window("hamming", 16, false), G(sg_hamming));
}

TEST_CASE("waveforms") {
  cv(sg::chirp(V(sg_tw), 2.0, 1.0, 8.0), G(sg_chirp));
  auto tw = tov(V(sg_tw));
  std::vector<double> phase(tw.size());
  for (size_t i = 0; i < tw.size(); ++i) phase[i] = 2 * M_PI * 3 * tw[i];
  auto ph = vec(phase.data(), (int)phase.size());
  cv(sg::sawtooth(ph), G(sg_sawtooth));
  cv(sg::square(ph, 0.3), G(sg_square));
}

TEST_CASE("filter design: butter / cheby") {
  cv(sg::butter(4, {0.2}, "low").b, G(sg_butter_b));
  cv(sg::butter(4, {0.2}, "low").a, G(sg_butter_a));
  cv(sg::butter(3, {0.3}, "high").b, G(sg_butterhp_b));
  cv(sg::butter(3, {0.3}, "high").a, G(sg_butterhp_a));
  cv(sg::butter(2, {0.2, 0.4}, "band").b, G(sg_butterbp_b), 1e-6, 1e-8);
  cv(sg::butter(2, {0.2, 0.4}, "band").a, G(sg_butterbp_a), 1e-6, 1e-8);
  cv(sg::cheby1(4, 1.0, {0.2}, "low").b, G(sg_cheby1_b), 1e-6, 1e-8);
  cv(sg::cheby1(4, 1.0, {0.2}, "low").a, G(sg_cheby1_a), 1e-6, 1e-8);
  cv(sg::cheby2(4, 30.0, {0.2}, "low").b, G(sg_cheby2_b), 1e-6, 1e-8);
  cv(sg::cheby2(4, 30.0, {0.2}, "low").a, G(sg_cheby2_a), 1e-6, 1e-8);
  cv(sg::firwin(21, {0.3}), G(sg_firwin), 1e-7, 1e-9);
}

TEST_CASE("filtering") {
  auto b = V(sg_butter_b), a = V(sg_butter_a), sig = V(sg_sig);
  cv(sg::lfilter(b, a, sig), G(sg_lfilter));
  cv(sg::lfilter_zi(b, a), G(sg_lfilter_zi));
  cv(sg::filtfilt(b, a, sig), G(sg_filtfilt), 1e-6, 1e-8);
  cv(sg::detrend(sig, "linear"), G(sg_detrend));
  // sosfilt with scipy-provided sos matches scipy's sosfilt
  numpp::ndarray sos(numpp::Shape{golden::sg_sos_r, golden::sg_sos_c}, numpp::kFloat64);
  { double* p = sos.typed_data<double>(); for (int i = 0; i < golden::sg_sos_r * golden::sg_sos_c; ++i) p[i] = golden::sg_sos_d[i]; }
  cv(sg::sosfilt(sos, sig), G(sg_sosfilt), 1e-6, 1e-8);
  // tf2sos produces a valid factorization (same response as lfilter). The SOS
  // coefficients come from numeric polynomial roots; Butterworth's 4-fold zero
  // at z=-1 is ill-conditioned, so engineering-precision agreement is expected.
  auto y_sos = tov(sg::sosfilt(sg::tf2sos(b, a), sig));
  auto y_lf = tov(sg::lfilter(b, a, sig));
  for (size_t i = 0; i < y_lf.size(); ++i) CHECK_CLOSE(y_sos[i], y_lf[i], 5e-3, 5e-4);
}

TEST_CASE("freqz and hilbert") {
  auto b = V(sg_butter_b), a = V(sg_butter_a);
  auto fz = sg::freqz(b, a, 32);
  cv(fz.w, G(sg_freqz_w));
  numpp::ndarray hc = fz.h.astype(numpp::kComplex128).ascontiguousarray();
  const std::complex<double>* h = hc.typed_data<std::complex<double>>();
  for (int i = 0; i < golden::sg_freqz_hr_n; ++i) {
    CHECK_CLOSE(h[i].real(), golden::sg_freqz_hr[i], R, A);
    CHECK_CLOSE(h[i].imag(), golden::sg_freqz_hi[i], R, A);
  }
  numpp::ndarray hi = sg::hilbert(V(sg_sig)).astype(numpp::kComplex128).ascontiguousarray();
  const std::complex<double>* hh = hi.typed_data<std::complex<double>>();
  for (int i = 0; i < golden::sg_hilbert_r_n; ++i) {
    CHECK_CLOSE(hh[i].real(), golden::sg_hilbert_r[i], R, A);
    CHECK_CLOSE(hh[i].imag(), golden::sg_hilbert_i[i], R, A);
  }
}

TEST_CASE("spectral estimation") {
  auto sig = V(sg_sig);
  auto per = sg::periodogram(sig, 50.0);
  cv(per.f, G(sg_per_f));
  cv(per.Pxx, G(sg_per_P), 1e-6, 1e-9);
  auto wel = sg::welch(sig, 50.0, "hann", 16);
  cv(wel.f, G(sg_welch_f));
  cv(wel.Pxx, G(sg_welch_P), 1e-6, 1e-9);
}
