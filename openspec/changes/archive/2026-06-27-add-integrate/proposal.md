# Add integrate + differentiate (Phase 5)

## Why

Phase 5 of the SciPP roadmap. `scipy.integrate` provides numerical quadrature and
ODE integration — the backbone of physics/engineering simulation — and
`scipy.differentiate` provides finite-difference derivatives. Both are pure
algorithm (like `optimize`), building on `numpp::ndarray` and `std::function`
callables.

`scipy.integrate` is large; this change delivers the workhorse explicit solvers
and fixed/adaptive quadrature, deferring the implicit (stiff) ODE methods and the
boundary-value solver, which need Newton/Jacobian machinery of their own.

## What changes

Adds the **integrate** capability — `scipp::integrate` and `scipp::differentiate`,
validated against the SciPy oracle:

- **Fixed-sample quadrature**: `trapezoid`, `simpson`, `cumulative_trapezoid`.
- **Adaptive quadrature**: `quad` (adaptive Gauss–Kronrod 21-point with interval
  subdivision, returning value and error estimate) and `fixed_quad`
  (Gauss–Legendre of fixed order).
- **ODE integration**: `solve_ivp` with the explicit Runge–Kutta methods
  `"RK45"` (Dormand–Prince 5(4)) and `"RK23"` (Bogacki–Shampine 3(2)), adaptive
  step-size control, and evaluation at requested `t_eval` points, returning an
  `OdeResult`.
- **Differentiation**: `scipp::differentiate::derivative` (central differences
  with Richardson extrapolation), `jacobian`, and `hessian`.

## Impact

- Affected specs: **adds** the `integrate` capability (covering both
  `scipp::integrate` and `scipp::differentiate`).
- Affected code: new `include/scipp/integrate/`, `include/scipp/differentiate/`,
  `src/integrate/`, `tests/test_integrate.cpp`, extended oracle generator. Reuses
  the Phase 1–4 foundation.
- Roadmap: checks off Phase 5 in `bootstrap-scipp-foundation/tasks.md`.

## Non-goals (deferred)

- **Implicit / stiff ODE solvers**: `Radau`, `BDF`, `LSODA`, and `odeint` (they
  require a Jacobian + Newton iteration).
- **Boundary-value problems**: `solve_bvp`.
- **Multidimensional quadrature**: `dblquad`, `tplquad`, `nquad`.
- **Other 1-D quadrature**: `romberg`, `quadrature`, `quad_vec`, QMC integration.
- **Dense-output interpolants** matching SciPy bit-for-bit: `solve_ivp` evaluates
  `t_eval` by stepping exactly onto each point rather than via SciPy's continuous
  extension; results are validated against analytic solutions.
