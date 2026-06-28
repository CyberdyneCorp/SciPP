# Design — integrate + differentiate (Phase 5)

## Context

`scipp::integrate` ports the explicit-solver core of `scipy.integrate`;
`scipp::differentiate` ports `scipy.differentiate`. Both are iterative numerical
algorithms over `numpp::ndarray` and `std::function` callables, reusing the
Phase-4 finite-difference style.

## API shape

```cpp
namespace scipp::integrate {
using Integrand = std::function<double(double)>;
using OdeFn     = std::function<ndarray(double, const ndarray&)>;   // (t, y) → dy/dt

struct QuadResult { double value; double abserr; };
struct OdeResult  { ndarray t; ndarray y; bool success; int nfev; std::string message; };
}
namespace scipp::differentiate {
struct DerivativeResult { double df; double error; bool success; };
}
```

`OdeResult.y` is shaped `(n_states, n_times)` like SciPy (`y[:, i]` is the state at
`t[i]`).

## Quadrature

- **trapezoid / simpson / cumulative_trapezoid** — fixed-sample rules over a `y`
  array with optional `x` or uniform `dx`; Simpson uses SciPy's even-interval
  correction. Exact match to SciPy.
- **fixed_quad** — Gauss–Legendre of order `n`: roots/weights from the Newton
  refinement of Legendre polynomials, mapped to `[a,b]`.
- **quad** — adaptive Gauss–Kronrod (G7,K15)-style using the **21-point Kronrod /
  10-point Gauss** rule per interval. The error estimate is
  `|K21 − G10|` (scaled as in QUADPACK). The worst-error interval is bisected from
  a heap until `epsabs`/`epsrel` is met or `limit` subdivisions are reached.
  Returns `(value, abserr)`. The node/weight tables are the standard GK21 constants.

## ODE integration (solve_ivp)

Explicit embedded Runge–Kutta with adaptive step control:

- **RK45** — Dormand–Prince 5(4): 7-stage tableau (FSAL), 5th-order solution with a
  4th-order embedded estimate.
- **RK23** — Bogacki–Shampine 3(2): 4-stage (FSAL).
- **Error control** — per-component scale `sc_i = atol + rtol·max(|y_i|, |y_new,i|)`,
  RMS error norm `‖err/sc‖`, accept if ≤ 1; step update
  `h ← h·clip(0.9·norm^(−1/(p+1)), 0.2, 5)` with the standard min/max factors.
- **Initial step** — SciPy's heuristic from the derivative scale, then adapted.
- **t_eval** — the step is capped so integration lands exactly on each requested
  point (no continuous-extension interpolation); avoids reproducing SciPy's dense
  output while keeping each sample at integration accuracy. Default `t_eval` is the
  two endpoints.

## differentiate

- **derivative(f, x)** — central differences with **Richardson extrapolation**
  (Romberg-style table over geometrically shrinking steps) to high accuracy, plus
  an error estimate from successive extrapolants.
- **jacobian(F, x)** — central-difference columns; **hessian(f, x)** — central
  second differences. Both return `numpp::ndarray`.

## Oracle strategy

- **Quadrature**: `quad`/`fixed_quad`/`simpson`/`trapezoid` compared `allclose` to
  SciPy on smooth integrands (polynomials, `exp`, `sin`, `1/(1+x²)`), with `quad`
  also checked against the analytic integral.
- **solve_ivp**: validated against **analytic solutions** (`y'=−y`, the harmonic
  oscillator, logistic growth) at `t_eval`, with `rtol=1e-8`/`atol=1e-10` and a
  comparison tolerance of ~1e-6 — sidestepping SciPy's interpolant while proving
  correctness and order.
- **differentiate**: `derivative`/`jacobian`/`hessian` compared to closed-form
  derivatives within ~1e-8.

## Open questions

- `solve_ivp` step capping at `t_eval` slightly changes the step sequence vs SciPy;
  since validation is against analytic solutions this is immaterial. If SciPy-exact
  sampling is later required, the RK dense-output interpolants can be added.
