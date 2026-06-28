# Add optimize (Phase 4)

## Why

Phase 4 of the SciPP roadmap. `scipy.optimize` is the first subpackage that is
**pure algorithm** — it does not delegate to NumPP's array kernels but implements
root finding, function minimization, and least-squares fitting on top of them.
It underpins `curve_fit`-style model fitting used across `stats`, `signal`, and
scientific applications.

`scipy.optimize` is large, so this change delivers the workhorse core and defers
the heavy/specialized solvers. The selected set covers the great majority of
real use: scalar/multivariate minimization, scalar/vector root finding, and
nonlinear least squares.

## What changes

Adds the **optimize** capability — `scipp::optimize`, validated against the SciPy
oracle, operating over `numpp::ndarray` and `std::function` callables:

- **Scalar root finding**: `brentq` (Brent bracketing), `bisect`, `newton`
  (Newton–Raphson with a secant fallback when no derivative is given).
- **Scalar minimization**: `minimize_scalar` with methods `"brent"` and
  `"bounded"`.
- **Multivariate minimization**: `minimize` with `"Nelder-Mead"` (derivative-free
  simplex) and `"BFGS"` (quasi-Newton with a finite-difference gradient and
  Armijo line search), returning a SciPy-shaped `OptimizeResult`.
- **Nonlinear least squares**: `least_squares` (Levenberg–Marquardt with a
  finite-difference Jacobian) and `curve_fit` (model fitting with parameter
  covariance), plus `fsolve` (multivariate Newton root finding).

## Impact

- Affected specs: **adds** the `optimize` capability.
- Affected code: new `include/scipp/optimize/`, `src/optimize/`,
  `tests/test_optimize.cpp`, extended oracle generator. Reuses the Phase 1–3
  foundation; uses `numpp::linalg::solve` for the LM/Newton linear systems.
- Roadmap: checks off Phase 4 in `bootstrap-scipp-foundation/tasks.md`.

## Non-goals (deferred to a later optimize change)

- **Linear & mixed-integer programming**: `linprog`, `milp` (simplex /
  interior-point / branch-and-bound) — their own change.
- **Constrained minimizers**: `SLSQP`, `trust-constr`, `COBYLA`, and bound/linear
  constraints beyond `minimize_scalar`'s `bounded`.
- **Global optimizers**: `differential_evolution`, `basinhopping`,
  `dual_annealing`, `shgo`.
- **Other `minimize` methods**: `Powell`, `CG`, `Newton-CG`, `L-BFGS-B`,
  `trust-*` (BFGS + Nelder-Mead cover the unconstrained core here).
- **`nnls`**, `root` with methods other than the default Newton, and
  analytic-Jacobian fast paths.
