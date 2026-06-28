// Window functions matching scipy.signal.windows (symmetric/periodic).
#include "scipp/signal/signal.hpp"

#include <cmath>
#include <vector>

#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"
#include "scipp/special/special.hpp"

namespace scipp::signal {
namespace {
namespace sd = scipp::linalg::detail;
constexpr double kPi = 3.141592653589793238462643383279502884;

// Periodic windows are the symmetric window of length M+1 with the last point
// dropped (scipy's `fftbins`/`_extend`).
int64_t ext_len(int64_t M, bool sym) { return sym ? M : M + 1; }
std::vector<double> trunc(std::vector<double> w, int64_t M, bool sym) {
  if (!sym) w.resize(M);
  return w;
}

std::vector<double> general_cosine(int64_t M, bool sym, const std::vector<double>& a) {
  if (M < 1) return {};
  if (M == 1) return {1.0};
  int64_t N = ext_len(M, sym);
  std::vector<double> w(N, 0.0);
  for (int64_t n = 0; n < N; ++n) {
    double fac = -kPi + 2.0 * kPi * n / (N - 1);
    double val = 0.0;
    for (size_t k = 0; k < a.size(); ++k) val += a[k] * std::cos(k * fac);
    w[n] = val;
  }
  return trunc(w, M, sym);
}
}  // namespace

ndarray boxcar(int64_t M, bool) { return sd::from_vec(std::vector<double>(M, 1.0)); }
ndarray hann(int64_t M, bool sym) { return sd::from_vec(general_cosine(M, sym, {0.5, 0.5})); }
ndarray hamming(int64_t M, bool sym) { return sd::from_vec(general_cosine(M, sym, {0.54, 0.46})); }
ndarray blackman(int64_t M, bool sym) {
  return sd::from_vec(general_cosine(M, sym, {0.42, 0.5, 0.08}));
}
ndarray blackmanharris(int64_t M, bool sym) {
  return sd::from_vec(general_cosine(M, sym, {0.35875, 0.48829, 0.14128, 0.01168}));
}
ndarray flattop(int64_t M, bool sym) {
  return sd::from_vec(general_cosine(
      M, sym, {0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368}));
}

ndarray bartlett(int64_t M, bool sym) {
  if (M < 1) return sd::from_vec({});
  if (M == 1) return sd::from_vec({1.0});
  int64_t N = ext_len(M, sym);
  std::vector<double> w(N);
  for (int64_t n = 0; n < N; ++n)
    w[n] = (n <= (N - 1) / 2.0) ? 2.0 * n / (N - 1) : 2.0 - 2.0 * n / (N - 1);
  return sd::from_vec(trunc(w, M, sym));
}

ndarray kaiser(int64_t M, double beta, bool sym) {
  if (M < 1) return sd::from_vec({});
  if (M == 1) return sd::from_vec({1.0});
  int64_t N = ext_len(M, sym);
  std::vector<double> w(N);
  double denom = special::i0(beta);
  for (int64_t n = 0; n < N; ++n) {
    double alpha = (N - 1) / 2.0;
    double r = (n - alpha) / alpha;
    w[n] = special::i0(beta * std::sqrt(1.0 - r * r)) / denom;
  }
  return sd::from_vec(trunc(w, M, sym));
}

ndarray tukey(int64_t M, double alpha, bool sym) {
  if (M < 1) return sd::from_vec({});
  if (M == 1) return sd::from_vec({1.0});
  if (alpha <= 0) return sd::from_vec(std::vector<double>(M, 1.0));
  int64_t N = ext_len(M, sym);
  if (alpha >= 1.0) return hann(M, sym);
  std::vector<double> w(N);
  int64_t width = static_cast<int64_t>(std::floor(alpha * (N - 1) / 2.0));
  for (int64_t n = 0; n < N; ++n) {
    if (n <= width)
      w[n] = 0.5 * (1.0 + std::cos(kPi * (-1.0 + 2.0 * n / alpha / (N - 1))));
    else if (n < N - width - 1)
      w[n] = 1.0;
    else
      w[n] = 0.5 * (1.0 + std::cos(kPi * (-2.0 / alpha + 1.0 + 2.0 * n / alpha / (N - 1))));
  }
  return sd::from_vec(trunc(w, M, sym));
}

ndarray get_window(const std::string& window, int64_t Nx, bool fftbins) {
  bool sym = !fftbins;
  if (window == "boxcar") return boxcar(Nx, sym);
  if (window == "hann" || window == "hanning") return hann(Nx, sym);
  if (window == "hamming") return hamming(Nx, sym);
  if (window == "blackman") return blackman(Nx, sym);
  if (window == "bartlett") return bartlett(Nx, sym);
  if (window == "blackmanharris") return blackmanharris(Nx, sym);
  if (window == "flattop") return flattop(Nx, sym);
  if (window == "tukey") return tukey(Nx, 0.5, sym);
  throw scipp::value_error("get_window: unknown window " + window);
}

}  // namespace scipp::signal
