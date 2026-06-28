// Discrete-time LTI systems: cont2discrete (zoh/bilinear/euler), dstep,
// dimpulse, dlsim, dfreqresp, dbode. Reuses the continuous helpers / expm.
#include "scipp/signal/signal.hpp"

#include <cmath>
#include <complex>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"
#include "scipp/linalg/linalg.hpp"

namespace scipp::signal {
namespace {
namespace sd = scipp::linalg::detail;
using cd = std::complex<double>;

// State-space dimensions and row-major buffers normalized from a system.
struct Unpacked {
  int n = 0, p = 0, q = 0;             // states, inputs, outputs
  std::vector<double> A, B, C, D;      // A: n*n, B: n*p, C: q*n, D: q*p
  double dt = 0.0;
};

// Read B as (n x p): 1-D treated as a single column (p = 1).
std::vector<double> b_as_mat(const ndarray& B, int n, int& p) {
  if (B.ndim() == 1) { p = 1; return sd::to_vec(B); }
  int64_t r, c; auto v = sd::to_mat(B, r, c); p = static_cast<int>(c); return v;
}
// Read C as (q x n): 1-D treated as a single row (q = 1).
std::vector<double> c_as_mat(const ndarray& C, int n, int& q) {
  if (C.ndim() == 1) { q = 1; return sd::to_vec(C); }
  int64_t r, c; auto v = sd::to_mat(C, r, c); q = static_cast<int>(r); return v;
}

Unpacked unpack(const ndarray& A, const ndarray& B, const ndarray& C,
                const ndarray& D, double dt) {
  Unpacked u; u.dt = dt;
  int64_t na, ca; u.A = sd::to_mat(A, na, ca); u.n = static_cast<int>(na);
  u.B = b_as_mat(B, u.n, u.p);
  u.C = c_as_mat(C, u.n, u.q);
  u.D = sd::to_vec(D);                 // q*p, row-major
  return u;
}

// C = A(n x k) * B(k x m), row-major.
std::vector<double> mm(const std::vector<double>& A, const std::vector<double>& B,
                       int n, int k, int m) {
  std::vector<double> C(static_cast<size_t>(n) * m, 0.0);
  for (int i = 0; i < n; ++i)
    for (int l = 0; l < k; ++l) {
      double a = A[i * k + l];
      for (int j = 0; j < m; ++j) C[i * m + j] += a * B[l * m + j];
    }
  return C;
}

// Solve M X = R for X, with M (n x n) and R (n x m). Uses numpp::linalg::solve.
std::vector<double> solve_mat(const std::vector<double>& M, const std::vector<double>& R,
                              int n, int m) {
  ndarray X = numpp::linalg::solve(sd::from_mat(M, n, n), sd::from_mat(R, n, m));
  return sd::to_vec(X);
}

std::vector<double> transpose(const std::vector<double>& A, int r, int c) {
  std::vector<double> T(static_cast<size_t>(r) * c);
  for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) T[j * r + i] = A[i * c + j];
  return T;
}

// alpha-parameterized linear discretization (Tustin alpha=0.5, Euler alpha=0).
DiscreteStateSpace alpha_discretize(const Unpacked& s, double alpha) {
  int n = s.n, p = s.p, q = s.q;
  std::vector<double> M(static_cast<size_t>(n) * n), Rp(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double id = (i == j) ? 1.0 : 0.0;
      M[i * n + j] = id - alpha * s.dt * s.A[i * n + j];
      Rp[i * n + j] = id + (1.0 - alpha) * s.dt * s.A[i * n + j];
    }
  std::vector<double> dtB(s.B.size());
  for (size_t i = 0; i < s.B.size(); ++i) dtB[i] = s.dt * s.B[i];

  std::vector<double> Ad = solve_mat(M, Rp, n, n);
  std::vector<double> Bd = solve_mat(M, dtB, n, p);
  // Cd = C * M^-1  == (M^T \ C^T)^T
  std::vector<double> Ct = transpose(s.C, q, n);
  std::vector<double> Cd = transpose(solve_mat(transpose(M, n, n), Ct, n, q), n, q);
  // Dd = D + alpha * C * Bd
  std::vector<double> CBd = mm(s.C, Bd, q, n, p);
  std::vector<double> Dd(static_cast<size_t>(q) * p);
  for (size_t i = 0; i < Dd.size(); ++i) Dd[i] = s.D[i] + alpha * CBd[i];

  return {sd::from_mat(Ad, n, n), sd::from_mat(Bd, n, p),
          sd::from_mat(Cd, q, n), sd::from_mat(Dd, q, p), s.dt};
}

DiscreteStateSpace zoh_discretize(const Unpacked& s) {
  int n = s.n, p = s.p, q = s.q, m = n + p;
  std::vector<double> Mb(static_cast<size_t>(m) * m, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) Mb[i * m + j] = s.A[i * n + j] * s.dt;
    for (int j = 0; j < p; ++j) Mb[i * m + (n + j)] = s.B[i * p + j] * s.dt;
  }
  std::vector<double> E = sd::to_vec(scipp::linalg::expm(sd::from_mat(Mb, m, m)));
  std::vector<double> Ad(static_cast<size_t>(n) * n), Bd(static_cast<size_t>(n) * p);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) Ad[i * n + j] = E[i * m + j];
    for (int j = 0; j < p; ++j) Bd[i * p + j] = E[i * m + (n + j)];
  }
  return {sd::from_mat(Ad, n, n), sd::from_mat(Bd, n, p),
          sd::from_mat(s.C, q, n), sd::from_mat(s.D, q, p), s.dt};
}

// Iterate x[k+1]=Ad x[k]+Bd u[k], y[k]=Cd x[k]+Dd u[k] for single input/output 0.
std::vector<double> simulate(const Unpacked& s, const std::vector<double>& u, int L) {
  int n = s.n;
  std::vector<double> x(n, 0.0), y(L, 0.0);
  for (int k = 0; k < L; ++k) {
    double yk = s.D.empty() ? 0.0 : s.D[0] * u[k];
    for (int i = 0; i < n; ++i) yk += s.C[i] * x[i];     // C row 0
    y[k] = yk;
    std::vector<double> xn(n, 0.0);
    for (int i = 0; i < n; ++i) {
      double v = s.B[i * s.p] * u[k];                     // input 0
      for (int j = 0; j < n; ++j) v += s.A[i * n + j] * x[j];
      xn[i] = v;
    }
    x = xn;
  }
  return y;
}

// Complex solve (zI - A) v = b via Gaussian elimination with partial pivoting.
std::vector<cd> csolve(std::vector<cd> M, std::vector<cd> b, int n) {
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::abs(M[r * n + col]) > std::abs(M[piv * n + col])) piv = r;
    if (piv != col) {
      for (int j = 0; j < n; ++j) std::swap(M[col * n + j], M[piv * n + j]);
      std::swap(b[col], b[piv]);
    }
    cd d = M[col * n + col];
    for (int r = col + 1; r < n; ++r) {
      cd f = M[r * n + col] / d;
      for (int j = col; j < n; ++j) M[r * n + j] -= f * M[col * n + j];
      b[r] -= f * b[col];
    }
  }
  std::vector<cd> x(n);
  for (int i = n - 1; i >= 0; --i) {
    cd s = b[i];
    for (int j = i + 1; j < n; ++j) s -= M[i * n + j] * x[j];
    x[i] = s / M[i * n + i];
  }
  return x;
}
}  // namespace

DiscreteStateSpace cont2discrete(const StateSpace& sys, double dt, const std::string& method) {
  Unpacked s = unpack(sys.A, sys.B, sys.C, sys.D, dt);
  if (method == "zoh") return zoh_discretize(s);
  if (method == "bilinear" || method == "tustin") return alpha_discretize(s, 0.5);
  if (method == "euler" || method == "forward_diff") return alpha_discretize(s, 0.0);
  if (method == "backward_diff") return alpha_discretize(s, 1.0);
  throw scipp::value_error("cont2discrete: unknown method '" + method + "'");
}

DiscreteStateSpace cont2discrete(const TransferFunction& sys, double dt, const std::string& method) {
  return cont2discrete(tf2ss(sys.num, sys.den), dt, method);
}

TimeResponse dstep(const DiscreteStateSpace& sys, int64_t n) {
  Unpacked s = unpack(sys.A, sys.B, sys.C, sys.D, sys.dt);
  int L = static_cast<int>(n);
  std::vector<double> u(L, 1.0), t(L);
  for (int k = 0; k < L; ++k) t[k] = k * sys.dt;
  return {sd::from_vec(t), sd::from_vec(simulate(s, u, L))};
}

TimeResponse dimpulse(const DiscreteStateSpace& sys, int64_t n) {
  Unpacked s = unpack(sys.A, sys.B, sys.C, sys.D, sys.dt);
  int L = static_cast<int>(n);
  std::vector<double> u(L, 0.0), t(L);
  if (L > 0) u[0] = 1.0;
  for (int k = 0; k < L; ++k) t[k] = k * sys.dt;
  return {sd::from_vec(t), sd::from_vec(simulate(s, u, L))};
}

TimeResponse dlsim(const DiscreteStateSpace& sys, const ndarray& u, const ndarray& t) {
  Unpacked s = unpack(sys.A, sys.B, sys.C, sys.D, sys.dt);
  std::vector<double> uv = sd::to_vec(u);
  int L = static_cast<int>(uv.size());
  std::vector<double> tv;
  if (t.ndim() >= 1 && t.size() == static_cast<int64_t>(L)) {
    tv = sd::to_vec(t);
  } else {
    tv.resize(L);
    for (int k = 0; k < L; ++k) tv[k] = k * sys.dt;
  }
  return {sd::from_vec(tv), sd::from_vec(simulate(s, uv, L))};
}

FreqRespResult dfreqresp(const DiscreteStateSpace& sys, const ndarray& w) {
  Unpacked s = unpack(sys.A, sys.B, sys.C, sys.D, sys.dt);
  std::vector<double> wv = sd::to_vec(w);
  int n = s.n;
  numpp::ndarray H(numpp::Shape{static_cast<int64_t>(wv.size())}, numpp::kComplex128);
  cd* h = H.typed_data<cd>();
  for (size_t idx = 0; idx < wv.size(); ++idx) {
    cd z = std::exp(cd(0.0, wv[idx]));
    std::vector<cd> M(static_cast<size_t>(n) * n), b(n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j)
        M[i * n + j] = (i == j ? z : cd(0.0)) - cd(s.A[i * n + j]);
      b[i] = cd(s.B[i * s.p]);                       // input 0
    }
    std::vector<cd> v = csolve(M, b, n);
    cd hv = s.D.empty() ? cd(0.0) : cd(s.D[0]);
    for (int i = 0; i < n; ++i) hv += cd(s.C[i]) * v[i];   // output 0
    h[idx] = hv;
  }
  return {w, H};
}

BodeResult dbode(const DiscreteStateSpace& sys, const ndarray& w) {
  FreqRespResult fr = dfreqresp(sys, w);
  numpp::ndarray Hc = fr.h.astype(numpp::kComplex128).ascontiguousarray();
  const cd* h = Hc.typed_data<cd>();
  int n = static_cast<int>(Hc.size());
  std::vector<double> mag(n), phase(n);
  for (int i = 0; i < n; ++i) {
    mag[i] = 20.0 * std::log10(std::abs(h[i]));
    phase[i] = std::arg(h[i]) * 180.0 / M_PI;
  }
  for (int i = 1; i < n; ++i) {
    double d = phase[i] - phase[i - 1];
    while (d > 180.0) { phase[i] -= 360.0; d = phase[i] - phase[i - 1]; }
    while (d < -180.0) { phase[i] += 360.0; d = phase[i] - phase[i - 1]; }
  }
  return {w, sd::from_vec(mag), sd::from_vec(phase)};
}

}  // namespace scipp::signal
