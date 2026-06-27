// FFT family — forward to numpp::fft (identical SciPy norm conventions) — plus
// helpers, next_fast_len, and the legacy fftpack re-exports.
#include "scypp/fft/fft.hpp"

#include "numpp/fft/fft.hpp"

namespace scypp::fft {

ndarray fft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return numpp::fft::fft(a, n, axis, norm);
}
ndarray ifft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return numpp::fft::ifft(a, n, axis, norm);
}
ndarray rfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return numpp::fft::rfft(a, n, axis, norm);
}
ndarray irfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return numpp::fft::irfft(a, n, axis, norm);
}
ndarray hfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return numpp::fft::hfft(a, n, axis, norm);
}
ndarray ihfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return numpp::fft::ihfft(a, n, axis, norm);
}

ndarray fft2(const ndarray& a, std::optional<std::vector<int64_t>> s, const std::string& norm) {
  return numpp::fft::fft2(a, s, std::nullopt, norm);
}
ndarray ifft2(const ndarray& a, std::optional<std::vector<int64_t>> s, const std::string& norm) {
  return numpp::fft::ifft2(a, s, std::nullopt, norm);
}
ndarray fftn(const ndarray& a, std::optional<std::vector<int64_t>> s, const std::string& norm) {
  return numpp::fft::fftn(a, s, std::nullopt, norm);
}
ndarray ifftn(const ndarray& a, std::optional<std::vector<int64_t>> s, const std::string& norm) {
  return numpp::fft::ifftn(a, s, std::nullopt, norm);
}
ndarray rfftn(const ndarray& a, std::optional<std::vector<int64_t>> s, const std::string& norm) {
  return numpp::fft::rfftn(a, s, std::nullopt, norm);
}
ndarray irfftn(const ndarray& a, std::optional<std::vector<int64_t>> s, const std::string& norm) {
  return numpp::fft::irfftn(a, s, std::nullopt, norm);
}

ndarray fftfreq(int64_t n, double d) { return numpp::fft::fftfreq(n, d); }
ndarray rfftfreq(int64_t n, double d) { return numpp::fft::rfftfreq(n, d); }
ndarray fftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes) {
  return numpp::fft::fftshift(a, axes);
}
ndarray ifftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes) {
  return numpp::fft::ifftshift(a, axes);
}

int64_t next_fast_len(int64_t n, bool /*real*/) {
  if (n <= 1) return 1;
  // Smallest 11-smooth integer >= n (factors only 2,3,5,7,11), as in SciPy.
  for (int64_t m = n;; ++m) {
    int64_t t = m;
    for (int p : {2, 3, 5, 7, 11})
      while (t % p == 0) t /= p;
    if (t == 1) return m;
  }
}

}  // namespace scypp::fft

namespace scypp::fftpack {
namespace f = scypp::fft;
ndarray fft(const ndarray& a, std::optional<int64_t> n, int64_t axis) { return f::fft(a, n, axis); }
ndarray ifft(const ndarray& a, std::optional<int64_t> n, int64_t axis) { return f::ifft(a, n, axis); }
ndarray rfft(const ndarray& a, std::optional<int64_t> n, int64_t axis) { return f::rfft(a, n, axis); }
ndarray irfft(const ndarray& a, std::optional<int64_t> n, int64_t axis) { return f::irfft(a, n, axis); }
ndarray dct(const ndarray& a, int type, int64_t axis) { return f::dct(a, type, axis); }
ndarray idct(const ndarray& a, int type, int64_t axis) { return f::idct(a, type, axis); }
ndarray dst(const ndarray& a, int type, int64_t axis) { return f::dst(a, type, axis); }
ndarray idst(const ndarray& a, int type, int64_t axis) { return f::idst(a, type, axis); }
}  // namespace scypp::fftpack
