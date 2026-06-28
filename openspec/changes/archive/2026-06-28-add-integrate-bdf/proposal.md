# Add BDF stiff ODE solver to solve_ivp

## Why

`solve_ivp` already ships the implicit `method="Radau"` for stiff systems, but
SciPy's other workhorse stiff integrator is `method="BDF"` (backward
differentiation formulas). BDF is the natural multistep companion to Radau and is
frequently the better choice on large, mildly stiff systems because each step
costs only a single Newton solve rather than a coupled multi-stage solve. It was
deferred from the stiff-solver backlog. Like Radau it needs only a dense linear
solve and a finite-difference Jacobian — both already exist in `solve_ivp.cpp` —
so no NumPP changes are required.

## What changes

Extends the **integrate** capability — `scipp::integrate::solve_ivp`, validated
against analytic stiff solutions:

- **`method="BDF"`**: implicit backward differentiation with adaptive step
  control, achieving **variable order 1–2** (implicit Euler / BDF1 plus the
  variable-step BDF2 formula). Each step uses a simplified Newton iteration with
  the Jacobian frozen at the step start (central-difference FD). The local error
  is estimated from the order-1 vs order-2 solutions (embedded estimate); the
  first step uses implicit-Euler step doubling with Richardson extrapolation. The
  method is A/L-stable on the stiff problems where the explicit RK methods stall.

## Impact

- Affected specs: **modifies** the `integrate` capability (adds one requirement).
- Affected code: extends `src/integrate/solve_ivp.cpp` (BDF Newton stage solve +
  adaptive order-1/2 driver, reusing the existing `eval`, `fd_jacobian`,
  `dense_solve`, `initial_step`, `rms_norm` helpers), `tests/test_integrate_bdf.cpp`,
  and the oracle generator. No header/signature change (new method string).
- Trims the BDF item from the `add-integrate-stiff-bvp` backlog.

## Non-goals

- Variable order **3–5** BDF (Nordsieck/backward-difference history with NDF
  coefficients). The delivered solver is fixed to orders 1–2, which matches the
  analytic stiff oracles to ~1e-5; higher orders remain in the backlog.
- `method="LSODA"` (stiff/non-stiff switching) and the legacy `odeint` entry
  point — separate backlog items.
- SciPy's exact step/order controller and dense-output interpolant; the adaptive
  order-1/2 controller validated here matches the true solution to tolerance,
  which is what the oracle checks.
