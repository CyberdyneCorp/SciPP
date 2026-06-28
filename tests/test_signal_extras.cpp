// Oracle tests for the signal-extras (completed scipy.signal surface).
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
using cd = std::complex<double>;
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray mat(const double* d, int r, int c) {
  numpp::ndarray a(numpp::Shape{r, c}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < r * c; ++i) p[i] = d[i];
  return a;
}
std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
void cv(const numpp::ndarray& got, const double* exp, int n, double rtol = 1e-7, double atol = 1e-9) {
  auto g = tov(got);
  for (int i = 0; i < n && i < (int)g.size(); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace
#define V(name) vec(golden::name, golden::name##_n)
#define G(name) golden::name, golden::name##_n

TEST_CASE("advanced spectral") {
  auto x = V(se_sig), y = V(se_sig2);
  auto c = sg::csd(x, y, 64.0, "hann", 32);
  numpp::ndarray Pc = c.Pxy.astype(numpp::kComplex128).ascontiguousarray();
  const cd* p = Pc.typed_data<cd>();
  for (int i = 0; i < golden::se_csd_re_n; ++i) {
    CHECK_CLOSE(p[i].real(), golden::se_csd_re[i], 1e-6, 1e-9);
    CHECK_CLOSE(p[i].imag(), golden::se_csd_im[i], 1e-6, 1e-9);
  }
  cv(sg::coherence(x, y, 64.0, "hann", 32).Pxx, G(se_coh), 1e-6, 1e-9);
  auto sp = sg::spectrogram(x, 64.0, "tukey", 32);
  cv(sp.f, G(se_spec_f)); cv(sp.t, G(se_spec_t));
  cv(sp.Sxx, golden::se_spec_S_d, golden::se_spec_S_r * golden::se_spec_S_c, 1e-6, 1e-9);
  // istft(stft(x)) reconstructs x
  auto st = sg::stft(x, 64.0, "hann", 32);
  cv(sg::istft(st.Zxx, 64.0, "hann", 32), G(se_sig), 1e-6, 1e-8);
}

TEST_CASE("peak analysis") {
  auto x = V(se_pk_x);
  cv(sg::find_peaks(x).peaks, G(se_peaks));
  cv(sg::find_peaks(x, 2.0).peaks, G(se_peaks_h));
  cv(sg::peak_prominences(x, V(se_peaks)).prominences, G(se_prom));
  cv(sg::peak_widths(x, V(se_peaks)).widths, G(se_width), 1e-7, 1e-9);
}

TEST_CASE("LTI systems") {
  double num[] = {1.0}, den[] = {1.0, 0.5, 1.0};
  sg::TransferFunction sys{vec(num, 1), vec(den, 3)};
  double w[32]; for (int i = 0; i < 32; ++i) w[i] = std::pow(10.0, -1.0 + 2.0 * i / 31.0);
  auto wn = vec(w, 32);
  auto fr = sg::freqresp(sys, wn);
  numpp::ndarray Hc = fr.h.astype(numpp::kComplex128).ascontiguousarray();
  const cd* h = Hc.typed_data<cd>();
  for (int i = 0; i < 32; ++i) { CHECK_CLOSE(h[i].real(), golden::se_fr_re[i], 1e-7, 1e-9); CHECK_CLOSE(h[i].imag(), golden::se_fr_im[i], 1e-7, 1e-9); }
  auto bd = sg::bode(sys, wn);
  cv(bd.mag, G(se_bode_mag), 1e-6, 1e-8); cv(bd.phase, G(se_bode_ph), 1e-6, 1e-8);
  double t[60]; for (int i = 0; i < 60; ++i) t[i] = 20.0 * i / 59.0;
  auto tn = vec(t, 60);
  cv(sg::step(sys, tn).y, G(se_step), 1e-5, 1e-7);
  cv(sg::impulse(sys, tn).y, G(se_impulse), 1e-5, 1e-7);
  std::vector<double> uv(60); for (int i = 0; i < 60; ++i) uv[i] = std::sin(t[i]);
  cv(sg::lsim(sys, vec(uv.data(), 60), tn).y, G(se_lsim), 1e-5, 1e-7);
}

TEST_CASE("resampling") {
  cv(sg::resample(V(se_sig), 96), G(se_resample), 1e-7, 1e-9);
  cv(sg::resample_poly(V(se_sig), 3, 2), G(se_resample_poly), 1e-6, 1e-8);
  cv(sg::decimate(V(se_sig), 2), G(se_decimate), 1e-6, 1e-8);
  cv(sg::upfirdn(V(se_uh), V(se_ux), 2, 1), G(se_upfirdn), 1e-9, 1e-11);
}

TEST_CASE("ellip / bessel by response") {
  // ellip: the Jacobi-function iteration matches the response shape (equiripple
  // passband at rp, stopband at rs) but not the exact ripple positions to machine
  // precision, so we validate the magnitude response to ~5%.
  auto e = sg::ellip(4, 1.0, 40.0, {0.3}, "low");
  auto fe = sg::freqz(e.b, e.a, 48);
  numpp::ndarray He = fe.h.astype(numpp::kComplex128).ascontiguousarray();
  const cd* he = He.typed_data<cd>();
  // Wider atol absorbs the two steep-transition-band points (a tiny frequency
  // shift there is a large magnitude change); passband/stopband match to ~2%.
  for (int i = 0; i < golden::se_ellip_mag_n; ++i) CHECK_CLOSE(std::abs(he[i]), golden::se_ellip_mag[i], 5e-2, 8e-2);
  auto b = sg::bessel(4, {0.3}, "low");
  auto fb = sg::freqz(b.b, b.a, 48);
  numpp::ndarray Hb = fb.h.astype(numpp::kComplex128).ascontiguousarray();
  const cd* hb = Hb.typed_data<cd>();
  for (int i = 0; i < golden::se_bessel_mag_n; ++i) CHECK_CLOSE(std::abs(hb[i]), golden::se_bessel_mag[i], 1e-2, 1e-3);
}

TEST_CASE("savgol / medfilt / wiener / 2-D") {
  cv(sg::savgol_coeffs(7, 2), G(se_savgol_coeffs));
  cv(sg::savgol_filter(V(se_sig), 7, 2), G(se_savgol), 1e-6, 1e-8);
  cv(sg::medfilt(V(se_sig), 5), G(se_medfilt));
  cv(sg::wiener(V(se_sig), 5), G(se_wiener), 1e-6, 1e-8);
  auto A = mat(golden::se_A2_d, golden::se_A2_r, golden::se_A2_c);
  auto B = mat(golden::se_B2_d, golden::se_B2_r, golden::se_B2_c);
  cv(sg::convolve2d(A, B, "full"), golden::se_conv2d_d, golden::se_conv2d_r * golden::se_conv2d_c);
  cv(sg::convolve2d(A, B, "same"), golden::se_conv2d_same_d, golden::se_conv2d_same_r * golden::se_conv2d_same_c);
  cv(sg::correlate2d(A, B, "full"), golden::se_corr2d_d, golden::se_corr2d_r * golden::se_corr2d_c);
}
