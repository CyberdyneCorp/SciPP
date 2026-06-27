// Advanced spectral: csd, coherence, spectrogram, stft, istft.
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

std::vector<cd> rfft_seg(const std::vector<double>& seg) {
  numpp::ndarray Z = numpp::fft::rfft(sd::from_vec(seg)).astype(numpp::kComplex128).ascontiguousarray();
  const cd* z = Z.typed_data<cd>();
  return std::vector<cd>(z, z + Z.size());
}

// Segment FFTs with constant detrend + window. Returns nseg columns of length nf.
std::vector<std::vector<cd>> segments(const std::vector<double>& x, const std::vector<double>& win,
                                      int nperseg, int noverlap) {
  int step = nperseg - noverlap;
  std::vector<std::vector<cd>> segs;
  for (int start = 0; start + nperseg <= static_cast<int>(x.size()); start += step) {
    std::vector<double> seg(nperseg);
    double mean = 0; for (int i = 0; i < nperseg; ++i) mean += x[start + i]; mean /= nperseg;
    for (int i = 0; i < nperseg; ++i) seg[i] = (x[start + i] - mean) * win[i];
    segs.push_back(rfft_seg(seg));
  }
  return segs;
}

void one_sided(std::vector<double>& P, int nperseg) {
  int nf = static_cast<int>(P.size());
  if (nperseg % 2 == 0) { for (int i = 1; i < nf - 1; ++i) P[i] *= 2.0; }
  else { for (int i = 1; i < nf; ++i) P[i] *= 2.0; }
}
std::vector<double> rfreq(int nperseg, double fs) {
  int nf = nperseg / 2 + 1;
  std::vector<double> f(nf);
  for (int i = 0; i < nf; ++i) f[i] = i * fs / nperseg;
  return f;
}
}  // namespace

CsdResult csd(const ndarray& x, const ndarray& y, double fs, const std::string& window,
              int64_t nperseg) {
  std::vector<double> xv = sd::to_vec(x), yv = sd::to_vec(y);
  int nps = static_cast<int>(std::min<int64_t>(nperseg, xv.size()));
  std::vector<double> win = sd::to_vec(get_window(window, nps, true));
  double winsq = 0; for (double w : win) winsq += w * w;
  double scale = 1.0 / (fs * winsq);
  auto sx = segments(xv, win, nps, nps / 2), sy = segments(yv, win, nps, nps / 2);
  int nf = nps / 2 + 1, nseg = static_cast<int>(sx.size());
  std::vector<cd> Pxy(nf, cd(0));
  for (int s = 0; s < nseg; ++s)
    for (int i = 0; i < nf; ++i) Pxy[i] += std::conj(sx[s][i]) * sy[s][i] * scale;
  std::vector<double> dummy(nf, 1.0);
  for (int i = 0; i < nf; ++i) Pxy[i] /= nseg;
  // one-sided doubling
  if (nps % 2 == 0) { for (int i = 1; i < nf - 1; ++i) Pxy[i] *= 2.0; }
  else { for (int i = 1; i < nf; ++i) Pxy[i] *= 2.0; }
  (void)dummy;
  numpp::ndarray P(numpp::Shape{nf}, numpp::kComplex128);
  cd* p = P.typed_data<cd>();
  for (int i = 0; i < nf; ++i) p[i] = Pxy[i];
  return {sd::from_vec(rfreq(nps, fs)), P};
}

SpectralResult coherence(const ndarray& x, const ndarray& y, double fs, const std::string& window,
                         int64_t nperseg) {
  SpectralResult pxx = welch(x, fs, window, nperseg);
  SpectralResult pyy = welch(y, fs, window, nperseg);
  CsdResult pxy = csd(x, y, fs, window, nperseg);
  std::vector<double> Pxx = sd::to_vec(pxx.Pxx), Pyy = sd::to_vec(pyy.Pxx);
  numpp::ndarray Pc = pxy.Pxy.astype(numpp::kComplex128).ascontiguousarray();
  const cd* p = Pc.typed_data<cd>();
  int nf = static_cast<int>(Pxx.size());
  std::vector<double> C(nf);
  for (int i = 0; i < nf; ++i) C[i] = std::norm(p[i]) / (Pxx[i] * Pyy[i]);
  return {pxx.f, sd::from_vec(C)};
}

SpectrogramResult spectrogram(const ndarray& x, double fs, const std::string&, int64_t nperseg) {
  std::vector<double> xv = sd::to_vec(x);
  int nps = static_cast<int>(std::min<int64_t>(nperseg, xv.size()));
  std::vector<double> win = sd::to_vec(tukey(nps, 0.25, /*sym=*/false));  // scipy default
  double winsq = 0; for (double w : win) winsq += w * w;
  double scale = 1.0 / (fs * winsq);
  int step = nps - nps / 8;  // noverlap default nperseg//8
  auto segs = segments(xv, win, nps, nps / 8);
  int nf = nps / 2 + 1, nseg = static_cast<int>(segs.size());
  std::vector<double> S(static_cast<size_t>(nf) * nseg);
  for (int s = 0; s < nseg; ++s) {
    std::vector<double> col(nf);
    for (int i = 0; i < nf; ++i) col[i] = std::norm(segs[s][i]) * scale;
    one_sided(col, nps);
    for (int i = 0; i < nf; ++i) S[i * nseg + s] = col[i];
  }
  std::vector<double> t(nseg);
  for (int s = 0; s < nseg; ++s) t[s] = (s * step + nps / 2.0) / fs;
  return {sd::from_vec(rfreq(nps, fs)), sd::from_vec(t), sd::from_mat(S, nf, nseg)};
}

StftResult stft(const ndarray& x, double fs, const std::string& window, int64_t nperseg) {
  std::vector<double> xv = sd::to_vec(x);
  int nps = static_cast<int>(std::min<int64_t>(nperseg, xv.size()));
  int nover = nps / 2, step = nps - nover;
  std::vector<double> win = sd::to_vec(get_window(window, nps, true));
  double winsum = 0; for (double w : win) winsum += w;
  double scale = 1.0 / winsum;
  std::vector<double> ext(nps / 2, 0.0);  // boundary='zeros'
  ext.insert(ext.end(), xv.begin(), xv.end());
  ext.insert(ext.end(), nps / 2, 0.0);
  int nadd = ((-(static_cast<int>(ext.size()) - nps)) % step + step) % step;  // padded
  ext.insert(ext.end(), nadd, 0.0);
  int nf = nps / 2 + 1;
  int nseg = (static_cast<int>(ext.size()) - nps) / step + 1;
  std::vector<cd> Z(static_cast<size_t>(nf) * nseg);
  for (int s = 0; s < nseg; ++s) {
    std::vector<double> seg(nps);
    for (int i = 0; i < nps; ++i) seg[i] = ext[s * step + i] * win[i];
    auto col = rfft_seg(seg);
    for (int i = 0; i < nf; ++i) Z[i * nseg + s] = col[i] * scale;
  }
  std::vector<double> t(nseg);
  for (int s = 0; s < nseg; ++s) t[s] = s * step / fs;
  numpp::ndarray Zxx(numpp::Shape{nf, nseg}, numpp::kComplex128);
  cd* zp = Zxx.typed_data<cd>();
  for (size_t i = 0; i < Z.size(); ++i) zp[i] = Z[i];
  return {sd::from_vec(rfreq(nps, fs)), sd::from_vec(t), Zxx};
}

ndarray istft(const ndarray& Zxx, double fs, const std::string& window, int64_t nperseg) {
  numpp::ndarray Z = Zxx.astype(numpp::kComplex128).ascontiguousarray();
  int nf = static_cast<int>(Z.shape()[0]), nseg = static_cast<int>(Z.shape()[1]);
  int nps = (nf - 1) * 2;
  (void)nperseg;
  int step = nps / 2;
  std::vector<double> win = sd::to_vec(get_window(window, nps, true));
  double winsum = 0; for (double w : win) winsum += w; double scale = 1.0 / winsum;
  const cd* zp = Z.typed_data<cd>();
  int total = (nseg - 1) * step + nps;
  std::vector<double> xr(total, 0.0), norm(total, 0.0);
  std::vector<cd> col(nf);
  for (int s = 0; s < nseg; ++s) {
    for (int i = 0; i < nf; ++i) col[i] = zp[i * nseg + s];
    numpp::ndarray ci(numpp::Shape{nf}, numpp::kComplex128);
    cd* cp = ci.typed_data<cd>(); for (int i = 0; i < nf; ++i) cp[i] = col[i];
    std::vector<double> seg = sd::to_vec(numpp::fft::irfft(ci, nps));  // = win*orig*scale
    for (int i = 0; i < nps; ++i) { xr[s * step + i] += seg[i] * win[i]; norm[s * step + i] += win[i] * win[i] * scale; }
  }
  std::vector<double> out;
  for (int i = nps / 2; i < total - nps / 2; ++i) out.push_back(norm[i] > 1e-12 ? xr[i] / norm[i] : 0.0);
  return sd::from_vec(out);
}

}  // namespace scypp::signal
