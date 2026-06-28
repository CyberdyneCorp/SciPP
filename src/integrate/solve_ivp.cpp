// Initial-value ODE integration: solve_ivp with explicit embedded Runge-Kutta
// methods RK45 (Dormand-Prince) and RK23 (Bogacki-Shampine), adaptive stepping.
#include "scypp/integrate/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::integrate {
namespace {
namespace sd = scypp::linalg::detail;

struct Tableau {
  int stages;
  std::vector<double> c;
  std::vector<std::vector<double>> A;  // stages x stages (lower triangular)
  std::vector<double> b;
  std::vector<double> E;  // error = b - b*
  int err_order;
};

Tableau rk45() {
  Tableau t;
  t.stages = 7;
  t.err_order = 4;
  t.c = {0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1};
  t.A.assign(7, std::vector<double>(7, 0.0));
  t.A[1][0] = 1.0 / 5;
  t.A[2][0] = 3.0 / 40; t.A[2][1] = 9.0 / 40;
  t.A[3][0] = 44.0 / 45; t.A[3][1] = -56.0 / 15; t.A[3][2] = 32.0 / 9;
  t.A[4][0] = 19372.0 / 6561; t.A[4][1] = -25360.0 / 2187; t.A[4][2] = 64448.0 / 6561; t.A[4][3] = -212.0 / 729;
  t.A[5][0] = 9017.0 / 3168; t.A[5][1] = -355.0 / 33; t.A[5][2] = 46732.0 / 5247; t.A[5][3] = 49.0 / 176; t.A[5][4] = -5103.0 / 18656;
  t.A[6][0] = 35.0 / 384; t.A[6][2] = 500.0 / 1113; t.A[6][3] = 125.0 / 192; t.A[6][4] = -2187.0 / 6784; t.A[6][5] = 11.0 / 84;
  t.b = {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0};
  t.E = {71.0 / 57600, 0, -71.0 / 16695, 71.0 / 1920, -17253.0 / 339200, 22.0 / 525, -1.0 / 40};
  return t;
}

Tableau rk23() {
  Tableau t;
  t.stages = 4;
  t.err_order = 2;
  t.c = {0, 1.0 / 2, 3.0 / 4, 1};
  t.A.assign(4, std::vector<double>(4, 0.0));
  t.A[1][0] = 1.0 / 2;
  t.A[2][1] = 3.0 / 4;
  t.A[3][0] = 2.0 / 9; t.A[3][1] = 1.0 / 3; t.A[3][2] = 4.0 / 9;
  t.b = {2.0 / 9, 1.0 / 3, 4.0 / 9, 0};
  t.E = {-5.0 / 72, 1.0 / 12, 1.0 / 9, -1.0 / 8};
  return t;
}

std::vector<double> eval(const OdeFn& f, double t, const std::vector<double>& y) {
  return sd::to_vec(f(t, sd::from_vec(y)));
}

double rms_norm(const std::vector<double>& e, const std::vector<double>& scale) {
  double s = 0.0;
  for (size_t i = 0; i < e.size(); ++i) { double r = e[i] / scale[i]; s += r * r; }
  return std::sqrt(s / e.size());
}

double initial_step(const OdeFn& f, double t0, const std::vector<double>& y0,
                    const std::vector<double>& f0, int order, double rtol, double atol, int& nfev) {
  int n = static_cast<int>(y0.size());
  std::vector<double> scale(n);
  for (int i = 0; i < n; ++i) scale[i] = atol + std::fabs(y0[i]) * rtol;
  double d0 = 0, d1 = 0;
  for (int i = 0; i < n; ++i) { double a = y0[i] / scale[i], b = f0[i] / scale[i]; d0 += a * a; d1 += b * b; }
  d0 = std::sqrt(d0 / n); d1 = std::sqrt(d1 / n);
  double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
  std::vector<double> y1(n);
  for (int i = 0; i < n; ++i) y1[i] = y0[i] + h0 * f0[i];
  std::vector<double> f1 = eval(f, t0 + h0, y1); ++nfev;
  double d2 = 0;
  for (int i = 0; i < n; ++i) { double r = (f1[i] - f0[i]) / scale[i]; d2 += r * r; }
  d2 = std::sqrt(d2 / n) / h0;
  double h1 = (std::max(d1, d2) <= 1e-15) ? std::max(1e-6, h0 * 1e-3)
                                          : std::pow(0.01 / std::max(d1, d2), 1.0 / (order + 1));
  return std::min(100.0 * h0, h1);
}

// ---- implicit (stiff) solver: Radau IIA, 3-stage order 5 ----

// Dense linear solve A x = b (m×m, row-major) via Gaussian elimination with
// partial pivoting. A and b are consumed; returns x.
std::vector<double> dense_solve(std::vector<double> A, std::vector<double> b, int m) {
  for (int col = 0; col < m; ++col) {
    int piv = col;
    for (int r = col + 1; r < m; ++r)
      if (std::fabs(A[r * m + col]) > std::fabs(A[piv * m + col])) piv = r;
    if (piv != col) {
      for (int j = 0; j < m; ++j) std::swap(A[col * m + j], A[piv * m + j]);
      std::swap(b[col], b[piv]);
    }
    double d = A[col * m + col];
    for (int r = col + 1; r < m; ++r) {
      double fct = A[r * m + col] / d;
      for (int j = col; j < m; ++j) A[r * m + j] -= fct * A[col * m + j];
      b[r] -= fct * b[col];
    }
  }
  std::vector<double> x(m);
  for (int r = m - 1; r >= 0; --r) {
    double s = b[r];
    for (int j = r + 1; j < m; ++j) s -= A[r * m + j] * x[j];
    x[r] = s / A[r * m + r];
  }
  return x;
}

// Central-difference Jacobian of f(t, y) at (t, y); returns n×n row-major.
std::vector<double> fd_jacobian(const OdeFn& f, double t, const std::vector<double>& y,
                                int& nfev) {
  int n = static_cast<int>(y.size());
  std::vector<double> J(n * n);
  std::vector<double> yp = y, ym = y;
  for (int j = 0; j < n; ++j) {
    double dh = 1e-8 * std::max(1.0, std::fabs(y[j]));
    yp[j] = y[j] + dh;
    ym[j] = y[j] - dh;
    auto fp = eval(f, t, yp);
    auto fm = eval(f, t, ym);
    nfev += 2;
    for (int i = 0; i < n; ++i) J[i * n + j] = (fp[i] - fm[i]) / (2.0 * dh);
    yp[j] = y[j];
    ym[j] = y[j];
  }
  return J;
}

// Radau IIA 3-stage coefficients (order 5, stiffly accurate: y_{n+1} = stage 3).
struct RadauTab {
  double c[3];
  double A[3][3];
  RadauTab() {
    double s6 = std::sqrt(6.0);
    c[0] = (4 - s6) / 10;
    c[1] = (4 + s6) / 10;
    c[2] = 1.0;
    A[0][0] = (88 - 7 * s6) / 360;
    A[0][1] = (296 - 169 * s6) / 1800;
    A[0][2] = (-2 + 3 * s6) / 225;
    A[1][0] = (296 + 169 * s6) / 1800;
    A[1][1] = (88 + 7 * s6) / 360;
    A[1][2] = (-2 - 3 * s6) / 225;
    A[2][0] = (16 - s6) / 36;
    A[2][1] = (16 + s6) / 36;
    A[2][2] = 1.0 / 9;
  }
};

// One Radau step from (t, y) of size h via simplified Newton (Jacobian frozen at
// the step start). Returns the new state; sets `ok` on Newton convergence.
std::vector<double> radau_step(const OdeFn& f, const RadauTab& T, double t, double h,
                               const std::vector<double>& y, int& nfev, bool& ok) {
  int n = static_cast<int>(y.size());
  int m = 3 * n;
  std::vector<double> J = fd_jacobian(f, t, y, nfev);
  // Newton matrix M = I_{3n} - h (A ⊗ J), constant across iterations.
  std::vector<double> M(m * m, 0.0);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k)
      for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
          double val = -h * T.A[i][k] * J[a * n + b];
          if (i == k && a == b) val += 1.0;
          M[(i * n + a) * m + (k * n + b)] = val;
        }
  std::vector<double> Z(m);  // stage values Y_i, initialised to y
  for (int i = 0; i < 3; ++i)
    for (int a = 0; a < n; ++a) Z[i * n + a] = y[a];

  ok = false;
  for (int it = 0; it < 20; ++it) {
    std::vector<std::vector<double>> F(3);
    for (int j = 0; j < 3; ++j) {
      std::vector<double> Yj(n);
      for (int a = 0; a < n; ++a) Yj[a] = Z[j * n + a];
      F[j] = eval(f, t + T.c[j] * h, Yj);
      ++nfev;
    }
    std::vector<double> G(m);
    for (int i = 0; i < 3; ++i)
      for (int a = 0; a < n; ++a) {
        double acc = 0.0;
        for (int j = 0; j < 3; ++j) acc += T.A[i][j] * F[j][a];
        G[i * n + a] = Z[i * n + a] - y[a] - h * acc;
      }
    std::vector<double> neg(m);
    for (int i = 0; i < m; ++i) neg[i] = -G[i];
    std::vector<double> dZ = dense_solve(M, neg, m);
    double dnorm = 0.0, znorm = 0.0;
    for (int i = 0; i < m; ++i) {
      Z[i] += dZ[i];
      dnorm += dZ[i] * dZ[i];
      znorm += Z[i] * Z[i];
    }
    if (std::sqrt(dnorm) <= 1e-10 * (1.0 + std::sqrt(znorm))) { ok = true; break; }
  }
  std::vector<double> ynew(n);
  for (int a = 0; a < n; ++a) ynew[a] = Z[2 * n + a];  // stiffly accurate
  return ynew;
}

OdeResult radau_integrate(const OdeFn& f, std::pair<double, double> t_span,
                          const ndarray& y0, std::optional<ndarray> t_eval,
                          double rtol, double atol) {
  RadauTab T;
  double t0 = t_span.first, tf = t_span.second;
  std::vector<double> y = sd::to_vec(y0);
  int n = static_cast<int>(y.size());
  int nfev = 0;

  std::vector<double> teval;
  if (t_eval) teval = sd::to_vec(*t_eval);
  else teval = {t0, tf};

  std::vector<std::vector<double>> ys;
  std::vector<double> ts;
  std::vector<double> fcur = eval(f, t0, y); ++nfev;
  double t = t0;
  double h = initial_step(f, t0, y, fcur, 5, rtol, atol, nfev);

  const double SAFETY = 0.9, MINF = 0.2, MAXF = 8.0, p = 5.0;
  const double R = std::pow(2.0, p) - 1.0;  // Richardson denominator
  size_t next = 0;
  if (!teval.empty() && teval[0] == t0) { ts.push_back(t0); ys.push_back(y); ++next; }
  bool success = true;
  int max_steps = 1000000, steps = 0;

  while (next < teval.size()) {
    double target = teval[next];
    while (t < target - 1e-15) {
      if (++steps > max_steps) { success = false; break; }
      double hcap = std::min(h, target - t);
      while (true) {
        bool ok1, ok2, ok3;
        auto yA = radau_step(f, T, t, hcap, y, nfev, ok1);          // one full step
        auto yh = radau_step(f, T, t, hcap / 2, y, nfev, ok2);      // two half steps
        auto yB = radau_step(f, T, t + hcap / 2, hcap / 2, yh, nfev, ok3);
        if (!(ok1 && ok2 && ok3)) {  // Newton failed -> shrink
          hcap *= 0.5;
          h = std::min(h, hcap);
          if (hcap < 1e-14) { success = false; break; }
          continue;
        }
        std::vector<double> err(n), scale(n), yext(n);
        for (int d = 0; d < n; ++d) {
          err[d] = (yB[d] - yA[d]) / R;
          yext[d] = yB[d] + err[d];  // local extrapolation
          scale[d] = atol + std::max(std::fabs(y[d]), std::fabs(yB[d])) * rtol;
        }
        double en = rms_norm(err, scale);
        if (en < 1.0) {
          double factor = (en == 0.0) ? MAXF : std::min(MAXF, SAFETY * std::pow(en, -1.0 / (p + 1)));
          t += hcap;
          y = yext;
          if (hcap == h) h *= factor;
          break;
        }
        hcap *= std::max(MINF, SAFETY * std::pow(en, -1.0 / (p + 1)));
        h = std::min(h, hcap);
      }
      if (!success) break;
    }
    if (!success) break;
    ts.push_back(target);
    ys.push_back(y);
    ++next;
  }

  OdeResult r;
  r.t = sd::from_vec(ts);
  ndarray Y(numpp::Shape{n, static_cast<int64_t>(ys.size())}, numpp::kFloat64);
  double* yp = Y.typed_data<double>();
  for (size_t j = 0; j < ys.size(); ++j)
    for (int i = 0; i < n; ++i) yp[i * static_cast<int64_t>(ys.size()) + j] = ys[j][i];
  r.y = Y;
  r.success = success;
  r.nfev = nfev;
  r.message = success ? "The solver successfully reached the end of the integration interval."
                      : "Maximum number of steps exceeded.";
  return r;
}

}  // namespace

OdeResult solve_ivp(const OdeFn& f, std::pair<double, double> t_span, const ndarray& y0,
                    const std::string& method, std::optional<ndarray> t_eval, double rtol,
                    double atol) {
  if (method == "Radau")
    return radau_integrate(f, t_span, y0, t_eval, rtol, atol);

  Tableau tab;
  if (method == "RK45") tab = rk45();
  else if (method == "RK23") tab = rk23();
  else throw scypp::value_error("solve_ivp: unknown method " + method);

  double t0 = t_span.first, tf = t_span.second;
  std::vector<double> y = sd::to_vec(y0);
  int n = static_cast<int>(y.size());
  int nfev = 0;

  std::vector<double> teval;
  if (t_eval) teval = sd::to_vec(*t_eval);
  else teval = {t0, tf};

  std::vector<std::vector<double>> ys;  // states at each eval point
  std::vector<double> ts;

  const double SAFETY = 0.9, MINF = 0.2, MAXF = 10.0;
  double err_exp = 1.0 / (tab.err_order + 1);

  std::vector<double> fcur = eval(f, t0, y); ++nfev;
  double t = t0;
  double h = initial_step(f, t0, y, fcur, tab.err_order, rtol, atol, nfev);

  size_t next = 0;
  if (!teval.empty() && teval[0] == t0) { ts.push_back(t0); ys.push_back(y); ++next; }

  std::vector<std::vector<double>> k(tab.stages, std::vector<double>(n));
  bool success = true;
  int max_steps = 1000000, steps = 0;

  while (next < teval.size()) {
    double target = teval[next];
    while (t < target - 1e-15) {
      if (++steps > max_steps) { success = false; break; }
      double hcap = std::min(h, target - t);  // cap so we land exactly on the eval point
      std::vector<double> ynew(n), yi(n), err(n);
      double err_norm;
      while (true) {
        k[0] = fcur;
        for (int i = 1; i < tab.stages; ++i) {
          for (int d = 0; d < n; ++d) {
            double acc = 0.0;
            for (int j = 0; j < i; ++j) acc += tab.A[i][j] * k[j][d];
            yi[d] = y[d] + hcap * acc;
          }
          k[i] = eval(f, t + tab.c[i] * hcap, yi); ++nfev;
        }
        for (int d = 0; d < n; ++d) {
          double acc = 0.0;
          for (int i = 0; i < tab.stages; ++i) acc += tab.b[i] * k[i][d];
          ynew[d] = y[d] + hcap * acc;
          double ea = 0.0;
          for (int i = 0; i < tab.stages; ++i) ea += tab.E[i] * k[i][d];
          err[d] = hcap * ea;
        }
        std::vector<double> scale(n);
        for (int d = 0; d < n; ++d) scale[d] = atol + std::max(std::fabs(y[d]), std::fabs(ynew[d])) * rtol;
        err_norm = rms_norm(err, scale);
        if (err_norm < 1.0) {
          double factor = (err_norm == 0.0) ? MAXF : std::min(MAXF, SAFETY * std::pow(err_norm, -err_exp));
          t += hcap;
          y = ynew;
          fcur = eval(f, t, y); ++nfev;  // (RK FSAL could reuse, kept simple)
          if (hcap == h) h *= factor;  // only grow when not capped
          break;
        } else {
          hcap *= std::max(MINF, SAFETY * std::pow(err_norm, -err_exp));
          h = std::min(h, hcap);
        }
      }
      if (!success) break;
    }
    if (!success) break;
    ts.push_back(target);
    ys.push_back(y);
    ++next;
  }

  OdeResult r;
  r.t = sd::from_vec(ts);
  ndarray Y(numpp::Shape{n, static_cast<int64_t>(ys.size())}, numpp::kFloat64);
  double* yp = Y.typed_data<double>();
  for (size_t j = 0; j < ys.size(); ++j)
    for (int i = 0; i < n; ++i) yp[i * static_cast<int64_t>(ys.size()) + j] = ys[j][i];
  r.y = Y;
  r.success = success;
  r.nfev = nfev;
  r.message = success ? "The solver successfully reached the end of the integration interval."
                      : "Maximum number of steps exceeded.";
  return r;
}

}  // namespace scypp::integrate
