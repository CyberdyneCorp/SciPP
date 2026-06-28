// Continuous-time LTI systems: tf2ss, freqresp, bode, lsim, step, impulse.
#include "scipp/signal/signal.hpp"

#include <cmath>
#include <complex>
#include <vector>

#include "numpp/backend/backend.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scipp/linalg/linalg.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::signal {
namespace {
namespace sd = scipp::linalg::detail;
using cd = std::complex<double>;

cd polyval(const std::vector<double>& c, cd s) {  // c descending powers
  cd r = 0;
  for (double ck : c) r = r * s + ck;
  return r;
}
}  // namespace

StateSpace tf2ss(const ndarray& num, const ndarray& den) {
  std::vector<double> b = sd::to_vec(num), a = sd::to_vec(den);
  double a0 = a[0];
  for (double& v : a) v /= a0;
  for (double& v : b) v /= a0;
  int N = static_cast<int>(a.size()) - 1;          // order
  std::vector<double> bp(N + 1, 0.0);              // left-pad num to N+1
  int off = (N + 1) - static_cast<int>(b.size());
  for (size_t i = 0; i < b.size(); ++i) bp[off + i] = b[i];

  std::vector<double> A(N * N, 0.0), B(N, 0.0), C(N, 0.0);
  for (int j = 0; j < N; ++j) A[0 * N + j] = -a[j + 1];
  for (int i = 1; i < N; ++i) A[i * N + (i - 1)] = 1.0;
  B[0] = 1.0;
  for (int j = 0; j < N; ++j) C[j] = bp[j + 1] - bp[0] * a[j + 1];
  std::vector<double> D{bp[0]};
  return {sd::from_mat(A, N, N), sd::from_vec(B), sd::from_vec(C), sd::from_vec(D)};
}

FreqRespResult freqresp(const TransferFunction& sys, const ndarray& w) {
  std::vector<double> b = sd::to_vec(sys.num), a = sd::to_vec(sys.den), wv = sd::to_vec(w);
  numpp::ndarray H(numpp::Shape{static_cast<int64_t>(wv.size())}, numpp::kComplex128);
  cd* h = H.typed_data<cd>();
  for (size_t i = 0; i < wv.size(); ++i) {
    cd s(0, wv[i]);
    h[i] = polyval(b, s) / polyval(a, s);
  }
  return {w, H};
}

BodeResult bode(const TransferFunction& sys, const ndarray& w) {
  FreqRespResult fr = freqresp(sys, w);
  numpp::ndarray Hc = fr.h.astype(numpp::kComplex128).ascontiguousarray();
  const cd* h = Hc.typed_data<cd>();
  int n = static_cast<int>(Hc.size());
  std::vector<double> mag(n), phase(n);
  for (int i = 0; i < n; ++i) { mag[i] = 20.0 * std::log10(std::abs(h[i])); phase[i] = std::arg(h[i]) * 180.0 / M_PI; }
  for (int i = 1; i < n; ++i) {  // unwrap
    double d = phase[i] - phase[i - 1];
    while (d > 180.0) { phase[i] -= 360.0; d = phase[i] - phase[i - 1]; }
    while (d < -180.0) { phase[i] += 360.0; d = phase[i] - phase[i - 1]; }
  }
  return {w, sd::from_vec(mag), sd::from_vec(phase)};
}

namespace {
std::vector<double> matvec(const std::vector<double>& M, const std::vector<double>& x, int n) {
  std::vector<double> y(n, 0.0);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) y[i] += M[i * n + j] * x[j];
  return y;
}
}  // namespace

namespace {
TimeResponse lsim_impl(const TransferFunction& sys, const ndarray& u, const ndarray& t, bool interp) {
  StateSpace ss = tf2ss(sys.num, sys.den);
  int n = static_cast<int>(ss.A.shape()[0]);
  std::vector<double> A = sd::to_vec(ss.A), B = sd::to_vec(ss.B), C = sd::to_vec(ss.C), D = sd::to_vec(ss.D);
  std::vector<double> uv = sd::to_vec(u), tv = sd::to_vec(t);
  int L = static_cast<int>(tv.size());
  double dt = tv[1] - tv[0];
  std::vector<double> Ad(n * n), Bd0(n), Bd1(n, 0.0);

  if (interp) {  // first-order hold: augmented matrix with a unit ramp entry
    int m = n + 2;
    std::vector<double> M(m * m, 0.0);
    for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) M[i * m + j] = A[i * n + j] * dt; M[i * m + n] = B[i] * dt; }
    M[n * m + (n + 1)] = 1.0;
    std::vector<double> E = sd::to_vec(scipp::linalg::expm(sd::from_mat(M, m, m)));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) Ad[i * n + j] = E[i * m + j];
      Bd1[i] = E[i * m + (n + 1)];          // ramp reconstruction:
      Bd0[i] = E[i * m + n] - Bd1[i];       // u(t)=u[i-1]+(u[i]-u[i-1]) t/dt
    }
  } else {  // zero-order hold
    int m = n + 1;
    std::vector<double> M(m * m, 0.0);
    for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) M[i * m + j] = A[i * n + j] * dt; M[i * m + n] = B[i] * dt; }
    std::vector<double> E = sd::to_vec(scipp::linalg::expm(sd::from_mat(M, m, m)));
    for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) Ad[i * n + j] = E[i * m + j]; Bd0[i] = E[i * m + n]; }
  }
  std::vector<double> x(n, 0.0), y(L);
  y[0] = D[0] * uv[0];
  for (int j = 0; j < n; ++j) y[0] += C[j] * x[j];
  for (int k = 1; k < L; ++k) {
    std::vector<double> xn = matvec(Ad, x, n);
    for (int i = 0; i < n; ++i) xn[i] += Bd0[i] * uv[k - 1] + Bd1[i] * uv[k];
    x = xn;
    double yk = D[0] * uv[k];
    for (int j = 0; j < n; ++j) yk += C[j] * x[j];
    y[k] = yk;
  }
  return {t, sd::from_vec(y)};
}
}  // namespace

TimeResponse lsim(const TransferFunction& sys, const ndarray& u, const ndarray& t) {
  return lsim_impl(sys, u, t, /*interp=*/true);  // scipy lsim default: first-order hold
}

TimeResponse step(const TransferFunction& sys, const ndarray& t) {
  std::vector<double> ones(sd::to_vec(t).size(), 1.0);
  return lsim_impl(sys, sd::from_vec(ones), t, /*interp=*/false);  // scipy step: ZOH
}

TimeResponse impulse(const TransferFunction& sys, const ndarray& t) {
  StateSpace ss = tf2ss(sys.num, sys.den);
  int n = static_cast<int>(ss.A.shape()[0]);
  std::vector<double> B = sd::to_vec(ss.B), C = sd::to_vec(ss.C);
  std::vector<double> tv = sd::to_vec(t);
  double dt = tv[1] - tv[0];
  std::vector<double> Adv(n * n);
  {
    std::vector<double> A = sd::to_vec(ss.A);
    for (double& v : A) v *= dt;
    Adv = sd::to_vec(scipp::linalg::expm(sd::from_mat(A, n, n)));
  }
  std::vector<double> x = B, y(tv.size());
  for (size_t k = 0; k < tv.size(); ++k) {
    double yk = 0; for (int j = 0; j < n; ++j) yk += C[j] * x[j];
    y[k] = yk;            // C @ expm(A*k*dt) @ B
    x = matvec(Adv, x, n);
  }
  return {t, sd::from_vec(y)};
}

}  // namespace scipp::signal
