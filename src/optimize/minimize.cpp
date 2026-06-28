// Multivariate minimization: Nelder-Mead simplex and BFGS quasi-Newton.
#include "scipp/optimize/optimize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "scipp/error.hpp"
#include "scipp/optimize/detail.hpp"

namespace scipp::optimize {
namespace {

using detail::tond;
using detail::tov;

using Vec = std::vector<double>;

double vdot(const Vec& a, const Vec& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}
double inf_norm(const Vec& a) {
  double m = 0.0;
  for (double v : a) m = std::max(m, std::fabs(v));
  return m;
}

// Central-difference gradient of an Rⁿ→R objective at x (more accurate than the
// forward difference used by BFGS; required by CG / L-BFGS-B).
Vec central_gradient(const ObjFn& f, const Vec& x, int& nfev) {
  int n = static_cast<int>(x.size());
  Vec g(n);
  for (int i = 0; i < n; ++i) {
    double h = detail::fd_step(x[i]);
    Vec xp = x, xm = x;
    xp[i] += h;
    xm[i] -= h;
    g[i] = (f(tond(xp)) - f(tond(xm))) / (2.0 * h);
    nfev += 2;
  }
  return g;
}

OptimizeResult nelder_mead(const ObjFn& f, std::vector<double> x0, double tol, int maxiter) {
  const double rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5;
  const double xatol = (tol > 0 ? tol : 1e-4), fatol = (tol > 0 ? tol : 1e-4);
  int n = static_cast<int>(x0.size());
  int nfev = 0;
  if (maxiter <= 0) maxiter = n * 200;

  std::vector<std::vector<double>> sim(n + 1, x0);
  for (int k = 0; k < n; ++k) {
    if (sim[k + 1][k] != 0.0) sim[k + 1][k] *= 1.05;
    else sim[k + 1][k] = 0.00025;
  }
  std::vector<double> fsim(n + 1);
  for (int i = 0; i <= n; ++i) { fsim[i] = f(tond(sim[i])); ++nfev; }

  auto order = [&]() {
    std::vector<int> idx(n + 1);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fsim[a] < fsim[b]; });
    std::vector<std::vector<double>> s2(n + 1);
    std::vector<double> f2(n + 1);
    for (int i = 0; i <= n; ++i) { s2[i] = sim[idx[i]]; f2[i] = fsim[idx[i]]; }
    sim = std::move(s2); fsim = std::move(f2);
  };
  order();

  auto axpy = [](double a, const std::vector<double>& x, double b, const std::vector<double>& y) {
    std::vector<double> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = a * x[i] + b * y[i];
    return r;
  };

  int it = 1;
  for (; it < maxiter; ++it) {
    double xspread = 0.0, fspread = 0.0;
    for (int i = 1; i <= n; ++i) {
      for (int j = 0; j < n; ++j) xspread = std::max(xspread, std::fabs(sim[i][j] - sim[0][j]));
      fspread = std::max(fspread, std::fabs(fsim[i] - fsim[0]));
    }
    if (xspread <= xatol && fspread <= fatol) break;

    std::vector<double> xbar(n, 0.0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) xbar[j] += sim[i][j] / n;

    std::vector<double> xr = axpy(1.0 + rho, xbar, -rho, sim[n]);
    double fxr = f(tond(xr)); ++nfev;
    bool shrink = false;
    if (fxr < fsim[0]) {
      std::vector<double> xe = axpy(1.0 + rho * chi, xbar, -rho * chi, sim[n]);
      double fxe = f(tond(xe)); ++nfev;
      if (fxe < fxr) { sim[n] = xe; fsim[n] = fxe; } else { sim[n] = xr; fsim[n] = fxr; }
    } else if (fxr < fsim[n - 1]) {
      sim[n] = xr; fsim[n] = fxr;
    } else if (fxr < fsim[n]) {
      std::vector<double> xc = axpy(1.0 + psi * rho, xbar, -psi * rho, sim[n]);
      double fxc = f(tond(xc)); ++nfev;
      if (fxc <= fxr) { sim[n] = xc; fsim[n] = fxc; } else shrink = true;
    } else {
      std::vector<double> xcc = axpy(1.0 - psi, xbar, psi, sim[n]);
      double fxcc = f(tond(xcc)); ++nfev;
      if (fxcc < fsim[n]) { sim[n] = xcc; fsim[n] = fxcc; } else shrink = true;
    }
    if (shrink) {
      for (int i = 1; i <= n; ++i) {
        sim[i] = axpy(1.0, sim[0], sigma, axpy(1.0, sim[i], -1.0, sim[0]));
        fsim[i] = f(tond(sim[i])); ++nfev;
      }
    }
    order();
  }
  OptimizeResult r;
  r.x = tond(sim[0]); r.fun = fsim[0]; r.success = it < maxiter; r.nit = it; r.nfev = nfev;
  r.message = r.success ? "Optimization terminated successfully." : "Maximum iterations reached.";
  return r;
}

OptimizeResult bfgs(const ObjFn& f, std::vector<double> x, double /*tol*/, int maxiter) {
  const double gtol = 1e-5, c1 = 1e-4;
  int n = static_cast<int>(x.size());
  int nfev = 0;
  std::vector<double> H(n * n, 0.0);
  for (int i = 0; i < n; ++i) H[i * n + i] = 1.0;
  std::vector<double> g = detail::num_gradient(f, x, nfev);
  double fx = f(tond(x)); ++nfev;

  auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s;
  };
  auto inf_norm = [&](const std::vector<double>& a) {
    double m = 0; for (double v : a) m = std::max(m, std::fabs(v)); return m;
  };

  int it = 0;
  for (; it < maxiter; ++it) {
    if (inf_norm(g) <= gtol) break;
    std::vector<double> p(n, 0.0);  // p = -H g
    for (int i = 0; i < n; ++i) { double s = 0; for (int j = 0; j < n; ++j) s += H[i * n + j] * g[j]; p[i] = -s; }
    double gp = dot(g, p);
    if (gp >= 0.0) {  // not a descent direction: reset to steepest descent
      for (int i = 0; i < n; ++i) p[i] = -g[i];
      gp = dot(g, p);
    }
    // Backtracking Armijo line search.
    double alpha = 1.0;
    std::vector<double> xn(n);
    double fxn;
    for (int ls = 0; ls < 60; ++ls) {
      for (int i = 0; i < n; ++i) xn[i] = x[i] + alpha * p[i];
      fxn = f(tond(xn)); ++nfev;
      if (fxn <= fx + c1 * alpha * gp) break;
      alpha *= 0.5;
    }
    std::vector<double> s(n), gn = detail::num_gradient(f, xn, nfev), y(n);
    for (int i = 0; i < n; ++i) { s[i] = xn[i] - x[i]; y[i] = gn[i] - g[i]; }
    double sy = dot(s, y);
    x = xn; fx = fxn; g = gn;
    if (sy > 1e-10) {  // BFGS inverse-Hessian update
      double rho = 1.0 / sy;
      std::vector<double> Hy(n, 0.0);
      for (int i = 0; i < n; ++i) { double t = 0; for (int j = 0; j < n; ++j) t += H[i * n + j] * y[j]; Hy[i] = t; }
      double yHy = dot(y, Hy);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          H[i * n + j] += (rho * rho * yHy + rho) * s[i] * s[j] - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
    }
  }
  OptimizeResult r;
  r.x = tond(x); r.fun = fx; r.success = inf_norm(g) <= gtol; r.nit = it; r.nfev = nfev;
  r.message = r.success ? "Optimization terminated successfully." : "Maximum iterations reached.";
  return r;
}

// ---------------------------------------------------------------------------
// Scalar 1-D minimizer (downhill bracket + Brent), used by Powell's line search.
// Mirrors scipy.optimize.bracket / scipy.optimize.Brent.
// ---------------------------------------------------------------------------
struct Bracket {
  double xa, xb, xc, fb;
};

Bracket bracket_min(const ScalarFn& f, int& nfev) {
  const double gold = 1.618034, tiny = 1e-21, grow = 110.0;
  double xa = 0.0, xb = 1.0;
  double fa = f(xa), fb = f(xb);
  nfev += 2;
  if (fa < fb) {
    std::swap(xa, xb);
    std::swap(fa, fb);
  }
  double xc = xb + gold * (xb - xa);
  double fc = f(xc);
  ++nfev;
  int iter = 0;
  while (fc < fb) {
    double t1 = (xb - xa) * (fb - fc), t2 = (xb - xc) * (fb - fa);
    double val = t2 - t1;
    double denom = (std::fabs(val) < tiny) ? 2.0 * tiny : 2.0 * val;
    double w = xb - ((xb - xc) * t2 - (xb - xa) * t1) / denom;
    double wlim = xb + grow * (xc - xb);
    double fw;
    if (++iter > 1000) break;
    if ((w - xc) * (xb - w) > 0.0) {  // w between xb and xc
      fw = f(w);
      ++nfev;
      if (fw < fc) {
        xa = xb; xb = w; fa = fb; fb = fw;
        break;
      } else if (fw > fb) {
        xc = w; fc = fw;
        break;
      }
      w = xc + gold * (xc - xb);
      fw = f(w);
      ++nfev;
    } else if ((w - wlim) * (wlim - xc) >= 0.0) {  // past the limit
      w = wlim;
      fw = f(w);
      ++nfev;
    } else if ((w - wlim) * (xc - w) > 0.0) {  // between xc and wlim
      fw = f(w);
      ++nfev;
      if (fw < fc) {
        xb = xc; xc = w; w = xc + gold * (xc - xb);
        fb = fc; fc = fw; fw = f(w);
        ++nfev;
      }
    } else {
      w = xc + gold * (xc - xb);
      fw = f(w);
      ++nfev;
    }
    xa = xb; xb = xc; xc = w;
    fa = fb; fb = fc; fc = fw;
  }
  return {xa, xb, xc, fb};
}

// Brent's parabolic-interpolation minimizer on [bracket]. Returns (xmin, fmin).
std::pair<double, double> brent_min(const ScalarFn& f, const Bracket& br, int& nfev,
                                    double tol = 1.48e-8, int maxiter = 500) {
  const double cg = 0.3819660, mintol = 1e-11;
  double x = br.xb, w = br.xb, v = br.xb;
  double fx = br.fb, fw = br.fb, fv = br.fb;
  double a = std::min(br.xa, br.xc), b = std::max(br.xa, br.xc);
  double deltax = 0.0, rat = 0.0;
  for (int it = 0; it < maxiter; ++it) {
    double xmid = 0.5 * (a + b);
    double tol1 = tol * std::fabs(x) + mintol, tol2 = 2.0 * tol1;
    if (std::fabs(x - xmid) < (tol2 - 0.5 * (b - a))) break;
    if (std::fabs(deltax) <= tol1) {
      deltax = (x >= xmid) ? (a - x) : (b - x);
      rat = cg * deltax;
    } else {
      double t1 = (x - w) * (fx - fv), t2 = (x - v) * (fx - fw);
      double p = (x - v) * t2 - (x - w) * t1;
      t2 = 2.0 * (t2 - t1);
      if (t2 > 0.0) p = -p;
      t2 = std::fabs(t2);
      double dx_temp = deltax;
      deltax = rat;
      if (p > t2 * (a - x) && p < t2 * (b - x) && std::fabs(p) < std::fabs(0.5 * t2 * dx_temp)) {
        rat = p / t2;
        double u = x + rat;
        if ((u - a) < tol2 || (b - u) < tol2) rat = (xmid - x >= 0) ? tol1 : -tol1;
      } else {
        deltax = (x >= xmid) ? (a - x) : (b - x);
        rat = cg * deltax;
      }
    }
    double u = (std::fabs(rat) < tol1) ? (x + (rat >= 0 ? tol1 : -tol1)) : (x + rat);
    double fu = f(u);
    ++nfev;
    if (fu > fx) {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; w = u; fv = fw; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
    } else {
      if (u >= x) a = x; else b = x;
      v = w; w = x; x = u; fv = fw; fw = fx; fx = fu;
    }
  }
  return {x, fx};
}

// Minimize f along x + alpha*d; updates x in place and returns the new f-value.
double line_search_powell(const ObjFn& f, Vec& x, const Vec& d, int& nfev) {
  int n = static_cast<int>(x.size());
  ScalarFn phi = [&](double alpha) {
    Vec xt(n);
    for (int i = 0; i < n; ++i) xt[i] = x[i] + alpha * d[i];
    return f(tond(xt));
  };
  Bracket br = bracket_min(phi, nfev);
  auto [alpha, fmin] = brent_min(phi, br, nfev);
  for (int i = 0; i < n; ++i) x[i] += alpha * d[i];
  return fmin;
}

OptimizeResult powell(const ObjFn& f, Vec x, double tol, int maxiter) {
  const double ftol = (tol > 0 ? tol : 1e-4), xtol = (tol > 0 ? tol : 1e-4);
  int n = static_cast<int>(x.size());
  int nfev = 0;
  if (maxiter <= 0) maxiter = n * 1000;
  std::vector<Vec> direc(n, Vec(n, 0.0));
  for (int i = 0; i < n; ++i) direc[i][i] = 1.0;

  double fval = f(tond(x));
  ++nfev;
  Vec x1 = x;
  int it = 0;
  for (; it < maxiter; ++it) {
    double fx = fval;
    int bigind = 0;
    double delta = 0.0;
    for (int i = 0; i < n; ++i) {
      double fx2 = fval;
      fval = line_search_powell(f, x, direc[i], nfev);
      if (fx2 - fval > delta) {
        delta = fx2 - fval;
        bigind = i;
      }
    }
    if (2.0 * (fx - fval) <= ftol * (std::fabs(fx) + std::fabs(fval)) + 1e-20) break;

    Vec direc1(n), x2(n);
    for (int i = 0; i < n; ++i) {
      direc1[i] = x[i] - x1[i];
      x2[i] = 2.0 * x[i] - x1[i];
    }
    x1 = x;
    double fx2 = f(tond(x2));
    ++nfev;
    if (fx > fx2) {
      double dt = 2.0 * (fx + fx2 - 2.0 * fval);
      double tmp = fx - fval - delta;
      dt *= tmp * tmp;
      tmp = fx - fx2;
      dt -= delta * tmp * tmp;
      if (dt < 0.0) {
        fval = line_search_powell(f, x, direc1, nfev);
        if (inf_norm(direc1) > 0.0) {
          direc[bigind] = direc[n - 1];
          direc[n - 1] = direc1;
        }
      }
    }
    // bound xtol via simplex-like spread of consecutive iterates
    double dx = 0.0;
    for (int i = 0; i < n; ++i) dx = std::max(dx, std::fabs(x[i] - x1[i]));
    (void)dx;  // ftol governs termination; xtol kept for API parity
    (void)xtol;
  }
  OptimizeResult r;
  r.x = tond(x);
  r.fun = fval;
  r.success = it < maxiter;
  r.nit = it;
  r.nfev = nfev;
  r.message = r.success ? "Optimization terminated successfully." : "Maximum iterations reached.";
  return r;
}

// ---------------------------------------------------------------------------
// Strong-Wolfe line search (Nocedal & Wright Alg. 3.5/3.6) over phi(a)=f(x+a p).
// Returns the accepted step length; *gnew holds the gradient at the new point.
// ---------------------------------------------------------------------------
double wolfe_line_search(const ObjFn& f, const Vec& x, const Vec& p, const Vec& g0, double f0,
                         Vec& gnew, double& fnew, int& nfev, double c1 = 1e-4, double c2 = 0.4) {
  int n = static_cast<int>(x.size());
  double dphi0 = vdot(g0, p);
  auto eval = [&](double a, Vec& gout, double& fout) {
    Vec xt(n);
    for (int i = 0; i < n; ++i) xt[i] = x[i] + a * p[i];
    fout = f(tond(xt));
    ++nfev;
    gout = central_gradient(f, xt, nfev);
    return vdot(gout, p);
  };

  auto zoom = [&](double alo, double ahi, double flo) {
    for (int j = 0; j < 60; ++j) {
      double aj = 0.5 * (alo + ahi);
      Vec gj;
      double fj;
      double dphij = eval(aj, gj, fj);
      if (fj > f0 + c1 * aj * dphi0 || fj >= flo) {
        ahi = aj;
      } else {
        if (std::fabs(dphij) <= -c2 * dphi0) {
          gnew = gj; fnew = fj;
          return aj;
        }
        if (dphij * (ahi - alo) >= 0.0) ahi = alo;
        alo = aj; flo = fj;
      }
    }
    Vec gj;
    double fj;
    eval(alo, gj, fj);
    gnew = gj; fnew = fj;
    return alo;
  };

  double a_prev = 0.0, f_prev = f0;
  double a = 1.0;
  for (int i = 1; i <= 30; ++i) {
    Vec gi;
    double fi;
    double dphii = eval(a, gi, fi);
    if (fi > f0 + c1 * a * dphi0 || (fi >= f_prev && i > 1)) {
      return zoom(a_prev, a, f_prev);
    }
    if (std::fabs(dphii) <= -c2 * dphi0) {
      gnew = gi; fnew = fi;
      return a;
    }
    if (dphii >= 0.0) return zoom(a, a_prev, fi);
    a_prev = a; f_prev = fi;
    a *= 2.0;
  }
  eval(a, gnew, fnew);  // budget exhausted: accept the last trial step
  return a;
}

OptimizeResult cg(const ObjFn& f, Vec x, double /*tol*/, int maxiter) {
  const double gtol = 1e-5;
  int n = static_cast<int>(x.size());
  int nfev = 0;
  if (maxiter <= 0) maxiter = n * 200;
  Vec g = central_gradient(f, x, nfev);
  double fx = f(tond(x));
  ++nfev;
  Vec p(n);
  for (int i = 0; i < n; ++i) p[i] = -g[i];
  int it = 0;
  for (; it < maxiter; ++it) {
    if (inf_norm(g) <= gtol) break;
    double deltak = vdot(g, g);
    Vec gn;
    double fn;
    double alpha = wolfe_line_search(f, x, p, g, fx, gn, fn, nfev);
    for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
    fx = fn;
    // Polak-Ribiere+ update
    Vec yk(n);
    for (int i = 0; i < n; ++i) yk[i] = gn[i] - g[i];
    double beta = deltak > 0.0 ? std::max(0.0, vdot(yk, gn) / deltak) : 0.0;
    for (int i = 0; i < n; ++i) p[i] = -gn[i] + beta * p[i];
    g = gn;
  }
  OptimizeResult r;
  r.x = tond(x);
  r.fun = fx;
  r.success = inf_norm(g) <= gtol;
  r.nit = it;
  r.nfev = nfev;
  r.message = r.success ? "Optimization terminated successfully." : "Maximum iterations reached.";
  return r;
}

// L-BFGS two-loop recursion: returns the search direction d = -H·g from the
// stored {s, y, rho} curvature pairs (newest last), with Nocedal's H0 scaling.
Vec lbfgs_direction(const Vec& g, const std::vector<Vec>& S, const std::vector<Vec>& Y,
                    const std::vector<double>& rho) {
  int n = static_cast<int>(g.size());
  int k = static_cast<int>(S.size());
  Vec q = g;
  std::vector<double> alpha(k);
  for (int i = k - 1; i >= 0; --i) {
    alpha[i] = rho[i] * vdot(S[i], q);
    for (int j = 0; j < n; ++j) q[j] -= alpha[i] * Y[i][j];
  }
  double gamma = 1.0;
  if (k > 0) {
    double sy = vdot(S[k - 1], Y[k - 1]), yy = vdot(Y[k - 1], Y[k - 1]);
    if (yy > 0.0) gamma = sy / yy;
  }
  Vec d(n);
  for (int j = 0; j < n; ++j) d[j] = gamma * q[j];
  for (int i = 0; i < k; ++i) {
    double beta = rho[i] * vdot(Y[i], d);
    for (int j = 0; j < n; ++j) d[j] += S[i][j] * (alpha[i] - beta);
  }
  for (int j = 0; j < n; ++j) d[j] = -d[j];
  return d;
}

// ---------------------------------------------------------------------------
// L-BFGS-B: limited-memory BFGS two-loop recursion with optional box bounds
// enforced by gradient projection (clamping the trial point to [lo, hi]).
// ---------------------------------------------------------------------------
OptimizeResult lbfgsb(const ObjFn& f, Vec x, double /*tol*/, int maxiter,
                      const std::optional<Bounds>& bounds) {
  const double pgtol = 1e-5, c1 = 1e-4;
  const int m = 10;
  int n = static_cast<int>(x.size());
  int nfev = 0;
  if (maxiter <= 0) maxiter = 15000;

  Vec lo(n, -std::numeric_limits<double>::infinity());
  Vec hi(n, std::numeric_limits<double>::infinity());
  if (bounds) {
    if (static_cast<int>(bounds->size()) != n)
      throw scipp::value_error("minimize: bounds length must match x0");
    for (int i = 0; i < n; ++i) {
      lo[i] = (*bounds)[i].first;
      hi[i] = (*bounds)[i].second;
    }
  }
  auto clip = [&](Vec v) {
    for (int i = 0; i < n; ++i) v[i] = std::min(std::max(v[i], lo[i]), hi[i]);
    return v;
  };
  // Projected-gradient infinity norm (the bound-aware stationarity measure).
  auto proj_grad_norm = [&](const Vec& xc, const Vec& g) {
    double m2 = 0.0;
    for (int i = 0; i < n; ++i) {
      double pg = xc[i] - std::min(std::max(xc[i] - g[i], lo[i]), hi[i]);
      m2 = std::max(m2, std::fabs(pg));
    }
    return m2;
  };

  x = clip(x);
  double fx = f(tond(x));
  ++nfev;
  Vec g = central_gradient(f, x, nfev);
  std::vector<Vec> S, Y;
  std::vector<double> rho;
  int it = 0;
  for (; it < maxiter; ++it) {
    if (proj_grad_norm(x, g) <= pgtol) break;
    Vec d = lbfgs_direction(g, S, Y, rho);

    // Backtracking Armijo on the projected trial point.
    double gd = vdot(g, d);
    if (gd >= 0.0) {  // not a descent direction: reset to projected steepest descent
      for (int j = 0; j < n; ++j) d[j] = -g[j];
      gd = vdot(g, d);
    }
    auto project = [&](double step) {
      Vec t(n);
      for (int j = 0; j < n; ++j) t[j] = x[j] + step * d[j];
      return clip(t);
    };
    double step = 1.0;
    Vec xn = project(step);
    double fn = f(tond(xn));
    ++nfev;
    for (int ls = 0; ls < 50; ++ls) {
      double rhs = 0.0;
      for (int j = 0; j < n; ++j) rhs += g[j] * (xn[j] - x[j]);
      if (fn <= fx + c1 * rhs) break;
      step *= 0.5;
      xn = project(step);
      fn = f(tond(xn));
      ++nfev;
    }
    Vec gn = central_gradient(f, xn, nfev);
    Vec s(n), y(n);
    for (int j = 0; j < n; ++j) {
      s[j] = xn[j] - x[j];
      y[j] = gn[j] - g[j];
    }
    double sy = vdot(s, y);
    if (sy > 1e-10) {
      S.push_back(s);
      Y.push_back(y);
      rho.push_back(1.0 / sy);
      if (static_cast<int>(S.size()) > m) {
        S.erase(S.begin());
        Y.erase(Y.begin());
        rho.erase(rho.begin());
      }
    }
    double xchange = 0.0;
    for (int j = 0; j < n; ++j) xchange = std::max(xchange, std::fabs(s[j]));
    x = xn; fx = fn; g = gn;
    if (xchange == 0.0) break;
  }
  OptimizeResult r;
  r.x = tond(x);
  r.fun = fx;
  r.success = proj_grad_norm(x, g) <= pgtol;
  r.nit = it;
  r.nfev = nfev;
  r.message = r.success ? "Optimization terminated successfully." : "Maximum iterations reached.";
  return r;
}

}  // namespace

OptimizeResult minimize(const ObjFn& f, const ndarray& x0, const std::string& method, double tol,
                        int maxiter, std::optional<Bounds> bounds) {
  std::vector<double> x = tov(x0);
  if (method == "Nelder-Mead") return nelder_mead(f, x, tol, maxiter);
  if (method == "BFGS") return bfgs(f, x, tol, maxiter);
  if (method == "Powell") return powell(f, x, tol, maxiter);
  if (method == "CG") return cg(f, x, tol, maxiter);
  if (method == "L-BFGS-B") return lbfgsb(f, x, tol, maxiter, bounds);
  throw scipp::value_error("minimize: unknown method " + method);
}

}  // namespace scipp::optimize
