// Two-point boundary value problems: solve_bvp via 4th-order Lobatto/Simpson
// collocation with a global Newton solve of the residual system on a fixed mesh.
//
// For y'(x) = f(x, y) on a mesh x[0..m-1] with boundary condition
// bc(y(x[0]), y(x[m-1])) = 0, each interval contributes the collocation residual
//
//   col_i = y_{i+1} - y_i - h/6 (f_i + 4 f_mid + f_{i+1}),
//   y_mid = 1/2 (y_i + y_{i+1}) - h/8 (f_{i+1} - f_i),  f_mid = f(x_mid, y_mid),
//
// which is exact through cubics (4th-order accurate). The unknowns are the node
// values y_{i,c}; the system has n*(m-1) collocation equations plus n boundary
// equations, i.e. n*m equations in n*m unknowns, solved by Newton iteration with
// a finite-difference Jacobian and dense Gaussian elimination. No adaptive mesh
// refinement (see Non-goals).
#include "scipp/integrate/integrate.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::integrate {
namespace {
namespace sd = scipp::linalg::detail;

// Dense linear solve A x = b (N×N, row-major) via Gaussian elimination with
// partial pivoting. A and b are consumed; returns x.
std::vector<double> dense_solve(std::vector<double> A, std::vector<double> b, int N) {
  for (int col = 0; col < N; ++col) {
    int piv = col;
    for (int r = col + 1; r < N; ++r)
      if (std::fabs(A[r * N + col]) > std::fabs(A[piv * N + col])) piv = r;
    if (piv != col) {
      for (int j = 0; j < N; ++j) std::swap(A[col * N + j], A[piv * N + j]);
      std::swap(b[col], b[piv]);
    }
    double d = A[col * N + col];
    if (d == 0.0) throw scipp::value_error("solve_bvp: singular Jacobian");
    for (int r = col + 1; r < N; ++r) {
      double fct = A[r * N + col] / d;
      for (int j = col; j < N; ++j) A[r * N + j] -= fct * A[col * N + j];
      b[r] -= fct * b[col];
    }
  }
  std::vector<double> x(N);
  for (int r = N - 1; r >= 0; --r) {
    double s = b[r];
    for (int j = r + 1; j < N; ++j) s -= A[r * N + j] * x[j];
    x[r] = s / A[r * N + r];
  }
  return x;
}

// Residual system R(u) of length n*m. Unknown layout: u[i*n + c] = y_c at node i.
// First n*(m-1) entries are interval collocation residuals; final n entries are
// the boundary-condition residuals.
struct Residual {
  const BvpFn& fun;
  const BcFn& bc;
  const std::vector<double>& xs;  // mesh, length m
  int n, m;

  ndarray node(const std::vector<double>& u, int i) const {
    ndarray y(numpp::Shape{n}, numpp::kFloat64);
    double* p = y.typed_data<double>();
    for (int c = 0; c < n; ++c) p[c] = u[i * n + c];
    return y;
  }

  std::vector<double> operator()(const std::vector<double>& u) const {
    std::vector<double> r(static_cast<size_t>(n) * m, 0.0);
    // Cache f at every node.
    std::vector<std::vector<double>> f(m);
    for (int i = 0; i < m; ++i) f[i] = sd::to_vec(fun(xs[i], node(u, i)));
    for (int i = 0; i < m - 1; ++i) {
      double h = xs[i + 1] - xs[i];
      ndarray ymid(numpp::Shape{n}, numpp::kFloat64);
      double* pm = ymid.typed_data<double>();
      for (int c = 0; c < n; ++c)
        pm[c] = 0.5 * (u[i * n + c] + u[(i + 1) * n + c]) - h / 8.0 * (f[i + 1][c] - f[i][c]);
      double xmid = 0.5 * (xs[i] + xs[i + 1]);
      std::vector<double> fmid = sd::to_vec(fun(xmid, ymid));
      for (int c = 0; c < n; ++c)
        r[i * n + c] = u[(i + 1) * n + c] - u[i * n + c]
                       - h / 6.0 * (f[i][c] + 4.0 * fmid[c] + f[i + 1][c]);
    }
    std::vector<double> bcr = sd::to_vec(bc(node(u, 0), node(u, m - 1)));
    if (static_cast<int>(bcr.size()) != n)
      throw scipp::value_error("solve_bvp: bc must return n residuals");
    for (int c = 0; c < n; ++c) r[(m - 1) * n + c] = bcr[c];
    return r;
  }
};

double max_abs(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s = std::max(s, std::fabs(x));
  return s;
}

}  // namespace

BvpResult solve_bvp(const BvpFn& fun, const BcFn& bc, const ndarray& x, const ndarray& y,
                    double tol, int max_iter) {
  std::vector<double> xs = sd::to_vec(x);
  int m = static_cast<int>(xs.size());
  if (m < 2) throw scipp::value_error("solve_bvp: mesh needs at least 2 nodes");
  if (y.ndim() != 2) throw scipp::value_error("solve_bvp: y guess must be 2-D (n, m)");
  numpp::ndarray yc = y.astype(numpp::kFloat64).ascontiguousarray();
  int n = static_cast<int>(yc.shape()[0]);
  if (static_cast<int>(yc.shape()[1]) != m)
    throw scipp::value_error("solve_bvp: y guess columns must match mesh length");

  // Flatten the guess into the unknown layout u[i*n + c] = y_c(x_i).
  const double* yp = yc.typed_data<double>();
  std::vector<double> u(static_cast<size_t>(n) * m);
  for (int c = 0; c < n; ++c)
    for (int i = 0; i < m; ++i) u[i * n + c] = yp[c * m + i];

  Residual R{fun, bc, xs, n, m};
  const int N = n * m;
  bool converged = false;

  for (int it = 0; it < max_iter; ++it) {
    std::vector<double> r0 = R(u);
    double rnorm = max_abs(r0);
    if (rnorm < tol) { converged = true; break; }

    // Finite-difference Jacobian (column-by-column), dense N×N row-major.
    std::vector<double> J(static_cast<size_t>(N) * N);
    for (int k = 0; k < N; ++k) {
      double save = u[k];
      double eps = 1e-7 * (1.0 + std::fabs(save));
      u[k] = save + eps;
      std::vector<double> rk = R(u);
      u[k] = save;
      for (int row = 0; row < N; ++row) J[row * N + k] = (rk[row] - r0[row]) / eps;
    }

    std::vector<double> delta = dense_solve(J, r0, N);  // J * delta = r0
    for (int k = 0; k < N; ++k) u[k] -= delta[k];

    if (max_abs(delta) < tol * (1.0 + max_abs(u))) {
      if (max_abs(R(u)) < std::max(tol, 1e-6)) { converged = true; break; }
    }
  }

  std::vector<double> rfinal = R(u);
  if (max_abs(rfinal) < std::max(tol, 1e-6)) converged = true;

  BvpResult out;
  out.x = sd::from_vec(xs);
  numpp::ndarray ys(numpp::Shape{n, m}, numpp::kFloat64);
  double* op = ys.typed_data<double>();
  for (int c = 0; c < n; ++c)
    for (int i = 0; i < m; ++i) op[c * m + i] = u[i * n + c];
  out.y = ys;
  out.success = converged;
  out.message = converged ? "The algorithm converged to the desired accuracy."
                          : "The Newton iteration did not converge.";
  return out;
}

}  // namespace scipp::integrate
