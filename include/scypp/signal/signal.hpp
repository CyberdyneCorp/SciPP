#pragma once
// scypp::signal — port of scipy.signal (Phase 8 subset): convolution, windows,
// waveforms, time-domain filtering, filter design, and spectral estimation.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scypp::signal {

using numpp::ndarray;

struct BA { ndarray b, a; };
struct Zpk { ndarray z, p; double k; };
struct FreqzResult { ndarray w, h; };          // h complex
struct SpectralResult { ndarray f, Pxx; };

// ---- convolution / correlation ----
ndarray convolve(const ndarray& a, const ndarray& v, const std::string& mode = "full");
ndarray correlate(const ndarray& a, const ndarray& v, const std::string& mode = "full");
ndarray fftconvolve(const ndarray& a, const ndarray& v, const std::string& mode = "full");

// ---- windows ----
ndarray boxcar(int64_t M, bool sym = true);
ndarray hann(int64_t M, bool sym = true);
ndarray hamming(int64_t M, bool sym = true);
ndarray blackman(int64_t M, bool sym = true);
ndarray bartlett(int64_t M, bool sym = true);
ndarray blackmanharris(int64_t M, bool sym = true);
ndarray flattop(int64_t M, bool sym = true);
ndarray kaiser(int64_t M, double beta, bool sym = true);
ndarray tukey(int64_t M, double alpha = 0.5, bool sym = true);
ndarray get_window(const std::string& window, int64_t Nx, bool fftbins = true);

// ---- waveforms ----
ndarray chirp(const ndarray& t, double f0, double t1, double f1,
              const std::string& method = "linear", double phi = 0.0);
ndarray sawtooth(const ndarray& t, double width = 1.0);
ndarray square(const ndarray& t, double duty = 0.5);
ndarray unit_impulse(int64_t n, int64_t idx = 0);

// ---- time-domain filtering ----
ndarray lfilter(const ndarray& b, const ndarray& a, const ndarray& x);
ndarray lfilter_zi(const ndarray& b, const ndarray& a);
ndarray filtfilt(const ndarray& b, const ndarray& a, const ndarray& x);
ndarray sosfilt(const ndarray& sos, const ndarray& x);
ndarray detrend(const ndarray& x, const std::string& type = "linear");
ndarray hilbert(const ndarray& x);   // analytic signal (complex)
FreqzResult freqz(const ndarray& b, const ndarray& a, int64_t worN = 512);

// ---- filter design ----
BA butter(int N, const std::vector<double>& Wn, const std::string& btype = "lowpass");
BA cheby1(int N, double rp, const std::vector<double>& Wn, const std::string& btype = "lowpass");
BA cheby2(int N, double rs, const std::vector<double>& Wn, const std::string& btype = "lowpass");
ndarray firwin(int numtaps, const std::vector<double>& cutoff, bool pass_zero = true,
               const std::string& window = "hamming", double fs = 2.0);
Zpk tf2zpk(const ndarray& b, const ndarray& a);
BA zpk2tf(const ndarray& z, const ndarray& p, double k);
ndarray zpk2sos(const ndarray& z, const ndarray& p, double k);
ndarray tf2sos(const ndarray& b, const ndarray& a);

// ---- spectral estimation ----
SpectralResult periodogram(const ndarray& x, double fs = 1.0, const std::string& window = "boxcar");
SpectralResult welch(const ndarray& x, double fs = 1.0, const std::string& window = "hann",
                     int64_t nperseg = 256);

// ---- advanced spectral (signal extras) ----
struct CsdResult { ndarray f, Pxy; };               // Pxy complex
SpectralResult coherence(const ndarray& x, const ndarray& y, double fs = 1.0,
                         const std::string& window = "hann", int64_t nperseg = 256);
CsdResult csd(const ndarray& x, const ndarray& y, double fs = 1.0,
              const std::string& window = "hann", int64_t nperseg = 256);
struct SpectrogramResult { ndarray f, t, Sxx; };    // Sxx (nf, nseg)
SpectrogramResult spectrogram(const ndarray& x, double fs = 1.0,
                              const std::string& window = "tukey", int64_t nperseg = 256);
struct StftResult { ndarray f, t, Zxx; };           // Zxx complex (nf, nseg)
StftResult stft(const ndarray& x, double fs = 1.0, const std::string& window = "hann",
                int64_t nperseg = 256);
ndarray istft(const ndarray& Zxx, double fs = 1.0, const std::string& window = "hann",
              int64_t nperseg = 256);

// ---- peak analysis ----
struct FindPeaksResult { ndarray peaks, prominences, widths, left_bases, right_bases; };
FindPeaksResult find_peaks(const ndarray& x, std::optional<double> height = std::nullopt,
                           std::optional<int64_t> distance = std::nullopt,
                           std::optional<double> prominence = std::nullopt,
                           std::optional<double> width = std::nullopt);
struct ProminencesResult { ndarray prominences, left_bases, right_bases; };
ProminencesResult peak_prominences(const ndarray& x, const ndarray& peaks);
struct WidthsResult { ndarray widths, width_heights, left_ips, right_ips; };
WidthsResult peak_widths(const ndarray& x, const ndarray& peaks, double rel_height = 0.5);

// ---- LTI systems (continuous-time transfer function) ----
struct TransferFunction { ndarray num, den; };
struct StateSpace { ndarray A, B, C, D; };
StateSpace tf2ss(const ndarray& num, const ndarray& den);
struct FreqRespResult { ndarray w, h; };            // h complex
FreqRespResult freqresp(const TransferFunction& sys, const ndarray& w);
struct BodeResult { ndarray w, mag, phase; };
BodeResult bode(const TransferFunction& sys, const ndarray& w);
struct TimeResponse { ndarray t, y; };
TimeResponse lsim(const TransferFunction& sys, const ndarray& u, const ndarray& t);
TimeResponse step(const TransferFunction& sys, const ndarray& t);
TimeResponse impulse(const TransferFunction& sys, const ndarray& t);

// ---- resampling ----
ndarray resample(const ndarray& x, int64_t num);
ndarray upfirdn(const ndarray& h, const ndarray& x, int up = 1, int down = 1);
ndarray resample_poly(const ndarray& x, int up, int down);
ndarray decimate(const ndarray& x, int q);

// ---- elliptic / Bessel design ----
BA ellip(int N, double rp, double rs, const std::vector<double>& Wn,
         const std::string& btype = "lowpass");
BA bessel(int N, const std::vector<double>& Wn, const std::string& btype = "lowpass");

// ---- Savitzky-Golay, median, Wiener, 2-D ----
ndarray savgol_coeffs(int window_length, int polyorder, int deriv = 0);
ndarray savgol_filter(const ndarray& x, int window_length, int polyorder, int deriv = 0);
ndarray medfilt(const ndarray& x, int kernel_size = 3);
ndarray wiener(const ndarray& x, int size = 3);
ndarray convolve2d(const ndarray& a, const ndarray& b, const std::string& mode = "full",
                   const std::string& boundary = "fill");
ndarray correlate2d(const ndarray& a, const ndarray& b, const std::string& mode = "full",
                    const std::string& boundary = "fill");

}  // namespace scypp::signal
