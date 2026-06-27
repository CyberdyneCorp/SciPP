// Convolution and correlation (full / same / valid).
#include "scypp/signal/signal.hpp"

#include <algorithm>
#include <vector>

#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;

std::vector<double> full_conv(const std::vector<double>& a, const std::vector<double>& v) {
  int n = static_cast<int>(a.size()), m = static_cast<int>(v.size());
  std::vector<double> out(n + m - 1, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) out[i + j] += a[i] * v[j];
  return out;
}

std::vector<double> crop(const std::vector<double>& full, int n, int m, const std::string& mode) {
  if (mode == "full") return full;
  if (mode == "same") {
    int outn = std::max(n, m);
    int start = (static_cast<int>(full.size()) - outn) / 2;
    return std::vector<double>(full.begin() + start, full.begin() + start + outn);
  }
  if (mode == "valid") {
    int outn = std::max(n, m) - std::min(n, m) + 1;
    int start = std::min(n, m) - 1;
    return std::vector<double>(full.begin() + start, full.begin() + start + outn);
  }
  throw scypp::value_error("convolve: unknown mode " + mode);
}
}  // namespace

ndarray convolve(const ndarray& a, const ndarray& v, const std::string& mode) {
  auto av = sd::to_vec(a), vv = sd::to_vec(v);
  return sd::from_vec(crop(full_conv(av, vv), av.size(), vv.size(), mode));
}

ndarray correlate(const ndarray& a, const ndarray& v, const std::string& mode) {
  auto av = sd::to_vec(a), vv = sd::to_vec(v);
  std::reverse(vv.begin(), vv.end());  // correlate(a,v) = convolve(a, reversed v)
  return sd::from_vec(crop(full_conv(av, vv), av.size(), vv.size(), mode));
}

ndarray fftconvolve(const ndarray& a, const ndarray& v, const std::string& mode) {
  // Direct convolution then crop — matches `convolve` to floating-point tolerance.
  auto av = sd::to_vec(a), vv = sd::to_vec(v);
  return sd::from_vec(crop(full_conv(av, vv), av.size(), vv.size(), mode));
}

}  // namespace scypp::signal
