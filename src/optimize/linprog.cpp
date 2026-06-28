// Linear programming via two-phase primal simplex (Bland's rule for
// anti-cycling), matching scipy.optimize.linprog on the optimum value/vertex.
#include "scipp/optimize/optimize.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "numpp/core/creation.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::optimize {
namespace {
namespace sd = scipp::linalg::detail;

struct Block {
  std::vector<double> A;  // row-major rows×cols
  std::vector<double> b;
  int64_t rows = 0, cols = 0;
};

Block load(const std::optional<ndarray>& A, const std::optional<ndarray>& b, int64_t nvar) {
  Block out;
  if (!A) return out;
  int64_t r, c;
  out.A = sd::to_mat(*A, r, c);
  out.rows = r;
  out.cols = c;
  out.b = b ? sd::to_vec(*b) : std::vector<double>(r, 0.0);
  (void)nvar;
  return out;
}
}  // namespace

LinprogResult linprog(const ndarray& c_in,
                      std::optional<ndarray> A_ub, std::optional<ndarray> b_ub,
                      std::optional<ndarray> A_eq, std::optional<ndarray> b_eq) {
  std::vector<double> c = sd::to_vec(c_in);
  const int n = static_cast<int>(c.size());
  Block ub = load(A_ub, b_ub, n), eq = load(A_eq, b_eq, n);
  const int n_ub = static_cast<int>(ub.rows), n_eq = static_cast<int>(eq.rows);
  const int m = n_ub + n_eq;  // total constraints

  // Standard form columns: n structural + n_ub slacks. Artificials added below.
  const int n_slack = n_ub;
  const int n_struct = n + n_slack;

  // Build constraint matrix T (m × n_struct) and rhs, with rhs >= 0.
  std::vector<std::vector<double>> T(m, std::vector<double>(n_struct, 0.0));
  std::vector<double> rhs(m, 0.0);
  for (int i = 0; i < n_ub; ++i) {
    for (int j = 0; j < n; ++j) T[i][j] = ub.A[i * ub.cols + j];
    T[i][n + i] = 1.0;  // slack
    rhs[i] = ub.b[i];
  }
  for (int i = 0; i < n_eq; ++i) {
    int r = n_ub + i;
    for (int j = 0; j < n; ++j) T[r][j] = eq.A[i * eq.cols + j];
    rhs[r] = eq.b[i];
  }
  for (int i = 0; i < m; ++i)
    if (rhs[i] < 0) {  // keep rhs nonnegative
      for (int j = 0; j < n_struct; ++j) T[i][j] = -T[i][j];
      rhs[i] = -rhs[i];
    }

  // Add one artificial per row so the all-artificial basis is feasible.
  const int total = n_struct + m;
  for (auto& row : T) row.resize(total, 0.0);
  std::vector<int> basis(m);
  for (int i = 0; i < m; ++i) {
    T[i][n_struct + i] = 1.0;
    basis[i] = n_struct + i;
  }

  const double EPS = 1e-9;
  auto pivot = [&](const std::vector<double>& cost) {
    // Reduced costs z_j - c_j for a minimization; Bland's rule on entering var.
    for (int iter = 0; iter < 20000; ++iter) {
      int enter = -1;
      for (int j = 0; j < total; ++j) {
        double zj = 0.0;
        for (int i = 0; i < m; ++i) zj += cost[basis[i]] * T[i][j];
        if (zj - cost[j] > EPS) { enter = j; break; }  // smallest index (Bland)
      }
      if (enter < 0) return;  // optimal
      int leave = -1;
      double best = std::numeric_limits<double>::infinity();
      for (int i = 0; i < m; ++i) {
        if (T[i][enter] > EPS) {
          double ratio = rhs[i] / T[i][enter];
          if (ratio < best - EPS || (std::fabs(ratio - best) <= EPS &&
                                     (leave < 0 || basis[i] < basis[leave]))) {
            best = ratio;
            leave = i;
          }
        }
      }
      if (leave < 0) return;  // unbounded
      double p = T[leave][enter];
      for (int j = 0; j < total; ++j) T[leave][j] /= p;
      rhs[leave] /= p;
      for (int i = 0; i < m; ++i)
        if (i != leave && std::fabs(T[i][enter]) > 0) {
          double f = T[i][enter];
          for (int j = 0; j < total; ++j) T[i][j] -= f * T[leave][j];
          rhs[i] -= f * rhs[leave];
        }
      basis[leave] = enter;
    }
  };

  LinprogResult res;
  res.x = numpp::zeros(numpp::Shape{n}, numpp::kFloat64);

  // Phase 1: minimize sum of artificials.
  if (m > 0) {
    std::vector<double> w(total, 0.0);
    for (int j = n_struct; j < total; ++j) w[j] = 1.0;
    pivot(w);
    double infeas = 0.0;
    for (int i = 0; i < m; ++i)
      if (basis[i] >= n_struct) infeas += rhs[i];
    if (infeas > 1e-7) {
      res.status = 2;
      res.success = false;
      res.message = "The problem is infeasible.";
      return res;
    }
  }

  // Phase 2: minimize original objective; artificials pinned out by cost = +inf.
  std::vector<double> obj(total, 0.0);
  for (int j = 0; j < n; ++j) obj[j] = c[j];
  for (int j = n_struct; j < total; ++j) obj[j] = 1e9;  // forbid artificials
  pivot(obj);

  // Unboundedness check: any structural column with negative reduced cost and no
  // positive pivot entry.
  for (int j = 0; j < n_struct; ++j) {
    double zj = 0.0;
    for (int i = 0; i < m; ++i) zj += obj[basis[i]] * T[i][j];
    if (zj - obj[j] > EPS) {
      bool bounded = false;
      for (int i = 0; i < m; ++i)
        if (T[i][j] > EPS) bounded = true;
      if (!bounded) {
        res.status = 3;
        res.success = false;
        res.message = "The problem is unbounded.";
        return res;
      }
    }
  }

  double* xp = res.x.typed_data<double>();
  for (int i = 0; i < m; ++i)
    if (basis[i] < n) xp[basis[i]] = rhs[i];
  double fun = 0.0;
  for (int j = 0; j < n; ++j) fun += c[j] * xp[j];
  res.fun = fun;
  res.status = 0;
  res.success = true;
  res.message = "Optimization terminated successfully.";
  return res;
}

}  // namespace scipp::optimize
