// Spectral estimation: periodogram and welch (density scaling, one-sided).
#include "scypp/signal/signal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/fft/fft.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;
using cd = std::complex<double>;

SpectralResult spectral(const std::vector<double>& x, double fs, const std::vector<double>& win,
                        int nperseg, int noverlap) {
  int step = nperseg - noverlap;
  int nf = nperseg / 2 + 1;
  double winsq = 0; for (double w : win) winsq += w * w;
  double scale = 1.0 / (fs * winsq);

  std::vector<double> Pxx(nf, 0.0);
  int nseg = 0;
  for (int start = 0; start + nperseg <= static_cast<int>(x.size()); start += step) {
    std::vector<double> seg(nperseg);
    double mean = 0;
    for (int i = 0; i < nperseg; ++i) mean += x[start + i];
    mean /= nperseg;
    for (int i = 0; i < nperseg; ++i) seg[i] = (x[start + i] - mean) * win[i];  // detrend constant + window
    numpp::ndarray Z = numpp::fft::rfft(sd::from_vec(seg));
    numpp::ndarray Zc = Z.astype(numpp::kComplex128).ascontiguousarray();
    const cd* z = Zc.typed_data<cd>();
    for (int i = 0; i < nf; ++i) Pxx[i] += std::norm(z[i]) * scale;
    ++nseg;
  }
  for (double& v : Pxx) v /= nseg;
  // one-sided doubling
  if (nperseg % 2 == 0) { for (int i = 1; i < nf - 1; ++i) Pxx[i] *= 2.0; }
  else { for (int i = 1; i < nf; ++i) Pxx[i] *= 2.0; }

  std::vector<double> f(nf);
  for (int i = 0; i < nf; ++i) f[i] = i * fs / nperseg;
  return {sd::from_vec(f), sd::from_vec(Pxx)};
}
}  // namespace

SpectralResult periodogram(const ndarray& x, double fs, const std::string& window) {
  std::vector<double> xv = sd::to_vec(x);
  int n = static_cast<int>(xv.size());
  std::vector<double> win = sd::to_vec(get_window(window, n, /*fftbins=*/true));
  return spectral(xv, fs, win, n, 0);
}

SpectralResult welch(const ndarray& x, double fs, const std::string& window, int64_t nperseg) {
  std::vector<double> xv = sd::to_vec(x);
  int n = static_cast<int>(xv.size());
  int nps = static_cast<int>(std::min<int64_t>(nperseg, n));
  std::vector<double> win = sd::to_vec(get_window(window, nps, /*fftbins=*/true));
  return spectral(xv, fs, win, nps, nps / 2);
}

}  // namespace scypp::signal
