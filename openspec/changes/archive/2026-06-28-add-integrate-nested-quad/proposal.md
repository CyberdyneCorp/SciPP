# Add nested and extended quadrature

## Why

`scipy.integrate` exposes several quadrature entry points beyond 1-D `quad` that
are widely used and were deferred from Phase 5: Romberg integration, vector-valued
`quad_vec`, and the nested adaptive integrators `dblquad`/`tplquad`/`nquad`. All of
them are pure compositions of the existing adaptive `quad` (Gauss-Kronrod) and the
trapezoidal rule — no NumPP changes — so they are low-risk, high-coverage drawdown.

## What changes

Extends the **integrate** capability — `scypp::integrate`, validated against the
SciPy oracle:

- **`romberg(f, a, b)`**: Richardson-extrapolated trapezoidal rule.
- **`quad_vec(f, a, b)`**: adaptive quadrature of a vector-valued integrand,
  integrating each component to tolerance.
- **`dblquad(f, a, b, gfun, hfun)`**: double integral with `x`-dependent inner
  bounds (`func(y, x)` convention, matching SciPy).
- **`tplquad(f, a, b, gfun, hfun, qfun, rfun)`**: triple integral with nested
  variable bounds (`func(z, y, x)` convention).
- **`nquad(f, ranges)`**: N-dimensional integral over a hyper-rectangle.

## Impact

- Affected specs: **modifies** the `integrate` capability (adds one requirement).
- Affected code: new `src/integrate/nested_quad.cpp`, new type aliases and decls
  in `include/scypp/integrate/integrate.hpp`, `tests/test_integrate_nested.cpp`,
  extended oracle generator. Built entirely on the existing `quad`.
- Trims the multidimensional-quadrature items from the `add-integrate-stiff-bvp`
  backlog.

## Non-goals

- Variable bounds in `nquad` (callable ranges with `args`) — fixed hyper-rectangle
  bounds only for now; `dblquad`/`tplquad` cover the variable-bound cases.
- `quad_vec`'s shared-evaluation-point optimisation (components are integrated
  independently); the returned value matches SciPy.
