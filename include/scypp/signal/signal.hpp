#pragma once
// scypp::signal — port of scipy.signal (Phase 8 subset): convolution, windows,
// waveforms, time-domain filtering, filter design, and spectral estimation.

#include <cstdint>
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

}  // namespace scypp::signal
