// Savitzky-Golay, median, Wiener, and 2-D convolution/correlation.
#include "scypp/signal/signal.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;

// Least-squares solve A c = y for the savgol/edge polynomial fits.
std::vector<double> lstsq_solve(const std::vector<double>& A, int rows, int cols,
                                const std::vector<double>& y) {
  numpp::linalg::LstsqResult r =
      numpp::linalg::lstsq(sd::from_mat(A, rows, cols), sd::from_vec(y));
  return sd::to_vec(r.solution);
}
}  // namespace

ndarray savgol_coeffs(int window_length, int polyorder, int deriv) {
  int halflen = window_length / 2;
  double pos = (window_length % 2 == 1) ? halflen : halflen - 0.5;
  std::vector<double> x(window_length);
  for (int i = 0; i < window_length; ++i) x[i] = -(- pos + i);  // arange(-pos, ...) reversed (conv)
  int po = polyorder + 1;
  std::vector<double> A(po * window_length);  // A[j][i] = x[i]^j
  for (int j = 0; j < po; ++j)
    for (int i = 0; i < window_length; ++i) A[j * window_length + i] = std::pow(x[i], j);
  std::vector<double> y(po, 0.0);
  double fact = 1.0; for (int i = 2; i <= deriv; ++i) fact *= i;
  y[deriv] = fact;
  return sd::from_vec(lstsq_solve(A, po, window_length, y));
}

ndarray savgol_filter(const ndarray& x, int window_length, int polyorder, int deriv) {
  std::vector<double> v = sd::to_vec(x);
  std::vector<double> c = sd::to_vec(savgol_coeffs(window_length, polyorder, deriv));
  int n = static_cast<int>(v.size()), h = window_length / 2;
  std::vector<double> y(n, 0.0);
  for (int i = h; i < n - h; ++i)
    for (int k = 0; k < window_length; ++k) y[i] += v[i - h + k] * c[k];
  // mode="interp": fit a polyorder polynomial to the edge windows.
  auto fit_edge = [&](int win_start, int eval_lo, int eval_hi) {
    int po = polyorder + 1;
    std::vector<double> A(window_length * po), b(window_length);
    for (int i = 0; i < window_length; ++i) {
      for (int j = 0; j < po; ++j) A[i * po + j] = std::pow(i, j);
      b[i] = v[win_start + i];
    }
    std::vector<double> coef = lstsq_solve(A, window_length, po, b);
    for (int idx = eval_lo; idx < eval_hi; ++idx) {
      double xp = idx - win_start, val = 0.0, fact = 1.0;
      for (int d = 0; d < deriv; ++d) fact *= 1;  // deriv handled below
      for (int j = deriv; j < po; ++j) {
        double term = coef[j];
        for (int r = 0; r < deriv; ++r) term *= (j - r);
        val += term * std::pow(xp, j - deriv);
      }
      y[idx] = val;
    }
  };
  if (n >= window_length) {
    fit_edge(0, 0, h);
    fit_edge(n - window_length, n - h, n);
  }
  return sd::from_vec(y);
}

ndarray medfilt(const ndarray& x, int kernel_size) {
  std::vector<double> v = sd::to_vec(x);
  int n = static_cast<int>(v.size()), h = kernel_size / 2;
  std::vector<double> y(n);
  std::vector<double> win(kernel_size);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < kernel_size; ++k) {
      int idx = i - h + k;
      win[k] = (idx >= 0 && idx < n) ? v[idx] : 0.0;  // zero-padded edges
    }
    std::sort(win.begin(), win.end());
    y[i] = win[kernel_size / 2];
  }
  return sd::from_vec(y);
}

ndarray wiener(const ndarray& x, int size) {
  std::vector<double> v = sd::to_vec(x);
  int n = static_cast<int>(v.size()), h = size / 2;
  std::vector<double> lmean(n, 0.0), lvar(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double m = 0, m2 = 0;
    for (int k = -h; k <= h; ++k) {
      int idx = i + k;
      double val = (idx >= 0 && idx < n) ? v[idx] : 0.0;
      m += val; m2 += val * val;
    }
    lmean[i] = m / size;
    lvar[i] = m2 / size - lmean[i] * lmean[i];
  }
  double noise = 0; for (double lv : lvar) noise += lv; noise /= n;
  std::vector<double> out(n);
  for (int i = 0; i < n; ++i) {
    if (lvar[i] < noise) out[i] = lmean[i];
    else out[i] = lmean[i] + (1.0 - noise / lvar[i]) * (v[i] - lmean[i]);
  }
  return sd::from_vec(out);
}

namespace {
ndarray conv2d_impl(const ndarray& a, const ndarray& b, const std::string& mode, bool correlate) {
  numpp::ndarray A = a.astype(numpp::kFloat64).ascontiguousarray();
  numpp::ndarray B = b.astype(numpp::kFloat64).ascontiguousarray();
  int M = static_cast<int>(A.shape()[0]), N = static_cast<int>(A.shape()[1]);
  int P = static_cast<int>(B.shape()[0]), Q = static_cast<int>(B.shape()[1]);
  const double* av = A.typed_data<double>();
  const double* bv = B.typed_data<double>();
  int FM = M + P - 1, FN = N + Q - 1;
  std::vector<double> full(static_cast<size_t>(FM) * FN, 0.0);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double aij = av[i * N + j];
      for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q) {
          // out[i+p][j+q] += A[i][j]*B[p][q] is convolution; correlation flips B.
          double bk = correlate ? bv[(P - 1 - p) * Q + (Q - 1 - q)] : bv[p * Q + q];
          full[(i + p) * FN + (j + q)] += aij * bk;
        }
    }
  int oM, oN, r0, c0;
  if (mode == "full") { oM = FM; oN = FN; r0 = c0 = 0; }
  else if (mode == "same") { oM = M; oN = N; r0 = (FM - M) / 2; c0 = (FN - N) / 2; }
  else if (mode == "valid") { oM = M - P + 1; oN = N - Q + 1; r0 = P - 1; c0 = Q - 1; }
  else throw scypp::value_error("convolve2d: unknown mode " + mode);
  std::vector<double> out(static_cast<size_t>(oM) * oN);
  for (int i = 0; i < oM; ++i)
    for (int j = 0; j < oN; ++j) out[i * oN + j] = full[(r0 + i) * FN + (c0 + j)];
  return sd::from_mat(out, oM, oN);
}
}  // namespace

ndarray convolve2d(const ndarray& a, const ndarray& b, const std::string& mode,
                   const std::string&) {
  return conv2d_impl(a, b, mode, /*correlate=*/false);
}
ndarray correlate2d(const ndarray& a, const ndarray& b, const std::string& mode,
                    const std::string&) {
  return conv2d_impl(a, b, mode, /*correlate=*/true);
}

}  // namespace scypp::signal
