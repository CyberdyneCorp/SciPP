# Add solve_bvp two-point boundary value problem solver

## Why

`scypp::integrate` covered initial-value problems (`solve_ivp`) and quadrature,
but not boundary value problems. SciPy exposes `solve_bvp` for systems
`y'(x) = f(x, y)` constrained by a two-point boundary condition `bc(ya, yb) = 0`,
which arise in shooting-free formulations of beam, diffusion, and eigenvalue
problems. It needs only a dense linear solve and a residual Newton iteration —
no NumPP changes — so it is a self-contained drawdown of the integrate backlog.

## What changes

Extends the **integrate** capability with a new entry point
`scypp::integrate::solve_bvp`, validated against analytic BVP solutions and
cross-checked against `scipy.integrate.solve_bvp`:

- **4th-order collocation**: each mesh interval uses the Lobatto/Simpson
  collocation residual `y_{i+1} - y_i - h/6 (f_i + 4 f_mid + f_{i+1})` with the
  internal midpoint state, exact through cubics.
- **Global Newton solve**: the `n*m` collocation + boundary residuals are solved
  with a finite-difference Jacobian and dense Gaussian elimination.
- New types `BvpFn`, `BcFn`, and `struct BvpResult{x, y, success, message}` in
  `include/scypp/integrate/integrate.hpp`.

## Impact

- Affected specs: **modifies** the `integrate` capability (adds one requirement).
- Affected code: new `src/integrate/solve_bvp.cpp` (wired into
  `src/CMakeLists.txt`), header types/signature, `tests/test_integrate_bvp.cpp`,
  extended oracle generator.
- Trims the `solve_bvp` items from the `add-integrate-stiff-bvp` backlog.

## Non-goals

- **No adaptive mesh refinement**: SciPy iteratively inserts mesh nodes to meet a
  residual tolerance; here the mesh supplied by the caller is fixed. A
  sufficiently fine mesh matches analytic solutions to ~1e-6 (and SciPy's own
  nodal values to ~1e-4), which is what the oracle validates. Adaptive refinement
  remains a backlog item.
- No singular-term / parameter-continuation (`p`) support, and no analytic
  Jacobian input — the Jacobian is formed by finite differences.
