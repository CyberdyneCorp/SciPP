# Add Radau stiff ODE solver to solve_ivp

## Why

`solve_ivp`'s explicit RK45/RK23 methods are unstable on stiff systems — they are
forced to take vanishingly small steps. SciPy's standard remedy is the implicit
`method="Radau"` (Radau IIA, 3-stage, order 5), which is A- and L-stable. It was
deferred from Phase 4. It needs only a dense linear solve and a finite-difference
Jacobian — no NumPP changes — so it is the highest-value stiff-solver drawdown.

## What changes

Extends the **integrate** capability — `scipp::integrate::solve_ivp`, validated
against the SciPy oracle:

- **`method="Radau"`**: implicit Radau IIA (3-stage, order 5, stiffly accurate)
  with a simplified Newton iteration (Jacobian frozen at the step start, computed
  by central differences) and Richardson step-doubling adaptive error control.
  Stable on stiff problems where the explicit methods stall.

## Impact

- Affected specs: **modifies** the `integrate` capability (adds one requirement).
- Affected code: extends `src/integrate/solve_ivp.cpp` (Radau tableau, dense
  solve, FD Jacobian, implicit driver), `tests/test_integrate_radau.cpp`,
  extended oracle generator. No header/signature change (new method string).
- Trims the Radau item from the `add-integrate-stiff-bvp` backlog.

## Non-goals

- `method="BDF"` (variable-order backward differentiation) and `method="LSODA"`
  (stiff/non-stiff switching) — remain in the backlog.
- The embedded Radau error estimator and complex-eigendecomposition stage solve
  (SciPy's optimisations); the simplified-Newton + Richardson approach matches the
  true solution to tolerance, which is what the oracle validates.
- `solve_bvp`, `odeint`, and multidimensional quadrature (separate backlog items).
